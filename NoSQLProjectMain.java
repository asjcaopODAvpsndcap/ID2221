package org.heima;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.types.StructType;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NoSQLProjectMain {

    private static final Logger logger = LoggerFactory.getLogger(NoSQLProjectMain.class);
    public static final String S3_BUCKET = "s3a://id2221";

    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
                .setAppName("NoSQLProject")
                .setMaster("local[*]")
                .set("spark.driver.memory", AppConfig.DEFAULT_DRIVER_MEMORY)
                .set("spark.executor.memory", AppConfig.DEFAULT_EXECUTOR_MEMORY)
                .set("spark.driver.maxResultSize", "2g")
                .set("spark.hadoop.fs.s3a.access.key", AppConfig.getAwsAccessKeyId())
                .set("spark.hadoop.fs.s3a.secret.key", AppConfig.getAwsSecretAccessKey())
                .set("spark.hadoop.fs.s3a.endpoint", AppConfig.getS3Endpoint())
                .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .set("spark.hadoop.fs.s3a.path.style.access", "true");

        SparkSession spark = SparkSession.builder()
                .config(conf)
                .getOrCreate();

        try {
            logger.info("开始处理PubMed XML数据...");

            // 1. 加载数据
            Dataset<Row> rawData = loadPubMedData(spark);
            logger.info("=== 原始数据检查 ===");
            logger.info("原始数据条数: {}", rawData.count());
            rawData.printSchema();
            rawData.show(10, false);

            // 2. 预处理数据
            Dataset<Row> processedData = preprocessData(spark, rawData);
            logger.info("=== 处理后数据检查 ===");
            logger.info("处理后数据条数: {}", processedData.count());

            if (processedData.count() == 0) {
                logger.error("⚠️ 警告：处理后没有数据！");
                return;
            }

            // 3. 显示示例数据
            logger.info("数据示例:");
            processedData.show(10, false);

            // 4. 按域分类统计
            logger.info("按医学领域分类:");
            processedData.groupBy("domain").count()
                    .orderBy(desc("count"))
                    .show();

            // 5. 保存处理后的数据到S3
            logger.info("保存数据到S3...");
            saveProcessedDataToS3(processedData);

            // 6. 演示查询功能
            logger.info("演示查询功能...");
            demonstrateQueries(spark);

        } catch (Exception e) {
            logger.error("处理过程中出现错误: {}", e.getMessage(), e);
        } finally {
            spark.stop();
        }
    }

    public static Dataset<Row> loadPubMedData(SparkSession spark) {
        logger.info("加载完整JATS格式论文XML（根标签：<article>）...");

        // 1. 先探索5条数据，验证结构（避免直接加载全部报错）
        Dataset<Row> exploratoryData = spark.read()
                .format("com.databricks.spark.xml")
                .option("rootTag", "article")  // ✅ 核心修复：根标签是<article>（不是PubmedArticleSet）
                .option("rowTag", "article")   // ✅ 行标签也是<article>（单篇论文对应1行数据）
                .option("inferSchema", "true")
                .option("treatEmptyValuesAsNulls", "true")
                .option("attributePrefix", "_")  // 读取标签属性（如ref的id）
                .option("ignoreSurroundingSpaces", "true")
                .load(S3_BUCKET + "/medical_xml_data/")  // 你的S3路径
                .limit(5);

        // 打印Schema和数据，确认字段存在（关键验证步骤）
        logger.info("=== 探索性数据Schema（确认front/body/back字段存在）===");
        exploratoryData.printSchema();
        logger.info("=== 探索性数据样例（确认摘要/标题字段非空）===");
        exploratoryData.select(
                "front.article-meta.article-id._VALUE",  // PMID
                "front.article-meta.title-group.article-title",  // 标题
                "front.article-meta.abstract.p"  // 摘要
        ).show(5, false);

        // 2. 加载全部数据（复用正确的根/行标签）
        return spark.read()
                .format("com.databricks.spark.xml")
                .option("rootTag", "article")
                .option("rowTag", "article")
                .option("inferSchema", "true")
                .option("treatEmptyValuesAsNulls", "true")
                .option("attributePrefix", "_")
                .option("ignoreSurroundingSpaces", "true")
                .load(S3_BUCKET + "/medical_xml_data/");
    }

    public static Dataset<Row> preprocessData(SparkSession spark, Dataset<Row> rawData) {
        logger.info("开始数据预处理（基于真实XML字段）...");

        long initialCount = rawData.count();
        logger.info("初始数据量: {}", initialCount);
        if (initialCount == 0) {
            logger.error("❌ 加载后无数据！请检查XML路径和rootTag");
            return rawData;
        }

        // ✅ 核心修复：不再一次性访问深层路径，而是分步、安全地提取
        // 我们将所有需要的结构先提取到顶层，如果不存在就设为 null
        Dataset<Row> step1 = rawData
                // 安全地提取 article-meta 和 journal-meta 结构体
                .withColumn("article_meta_struct", when(col("front").isNotNull(), col("front.article-meta")).otherwise(lit(null)))
                .withColumn("journal_meta_struct", when(col("front").isNotNull(), col("front.journal-meta")).otherwise(lit(null)));

        // 现在，基于这些安全的顶层结构体进行下一步提取
        Dataset<Row> step2 = step1
                // 1. 提取PMID (从 article_meta_struct 中提取)
                .withColumn("pmid",
                        when(col("article_meta_struct.article-id").isNotNull()
                                        .and(size(col("article_meta_struct.article-id")).gt(0)),
                                expr("filter(article_meta_struct.`article-id`, x -> x.`_pub-id-type` == 'pmid')[0].`_VALUE`")
                        ).otherwise(lit("unknown")))

                // 2. 提取文章标题 (从 article_meta_struct 中提取)
                .withColumn("article_title",
                        when(col("article_meta_struct.title-group.article-title").isNotNull(),
                                col("article_meta_struct.title-group.article-title").cast("string"))
                                .otherwise(lit("unknown")))

                // 3. 提取期刊名 (从 journal_meta_struct 中提取)
                .withColumn("journal",
                        when(col("journal_meta_struct.journal-title-group.journal-title").isNotNull(),
                                col("journal_meta_struct.journal-title-group.journal-title").cast("string"))
                                .otherwise(lit("unknown")))

                // 4. 提取发表年份 (从 article_meta_struct 中提取)
                .withColumn("year",
                        when(col("article_meta_struct.pub-date").isNotNull(),
                                expr("filter(article_meta_struct.`pub-date`, x -> x.`_pub-type` == 'ppub')[0].year")
                        ).otherwise(lit("unknown")))

                // 5. 提取内容 (从 article_meta_struct 和 body 中提取)
                .withColumn("abstract_text",
                        when(col("article_meta_struct.abstract.p").isNotNull(),
                                // 先用 flatten 将 ARRAY<ARRAY<STRING>> 降维成 ARRAY<STRING>
                                // 然后再用 concat_ws 拼接
                                concat_ws(" ", flatten(col("article_meta_struct.abstract.p"))))
                                .otherwise(lit("")))


                .withColumn("body_text",
                        when(col("body").isNotNull(),
                                col("body").cast("string"))
                                .otherwise(lit("")))
                .withColumn("citation_text",
                        concat_ws(" ", col("abstract_text"), col("body_text")))

                // 6. 生成唯一引用ID
                .withColumn("ref_id",
                        when(input_file_name().isNotNull(), input_file_name())
                                .otherwise(monotonically_increasing_id().cast("string")));

        // 打印提取结果
        logger.info("=== 字段提取示例 ===");
        step2.select("pmid", "article_title", "journal", "year", "citation_text")
                .show(5, false);

        // 内容过滤
        Dataset<Row> step3 = step2
                .filter(col("citation_text").isNotNull().and(length(trim(col("citation_text"))).gt(50)))
                .filter(col("pmid").isNotNull().and(col("pmid").notEqual("unknown")));

        long afterContentFilterCount = step3.count();
        logger.info("内容过滤后数据量: {} (因内容不合格过滤掉: {})",
                afterContentFilterCount, initialCount - afterContentFilterCount);

        if (afterContentFilterCount == 0) {
            logger.error("⚠️ 内容过滤后无数据！可能是所有记录的摘要或正文都为空或过短。");
            return step2.select( // 返回未经过内容过滤的数据，但只选择需要的字段
                    col("pmid"), col("article_title").alias("title"), col("journal"),
                    col("year"), col("citation_text").alias("content"), col("ref_id")
                    // ... 添加其他需要的字段 ...
            );
        }

        // 最终数据清理和选择
        return step3
                .withColumn("text_id", monotonically_increasing_id())
                .withColumn("content", trim(regexp_replace(regexp_replace(col("citation_text"), "<[^>]+>", " "), "[\\r\\n\\s]+", " ")))
                .withColumn("domain", classifyMedicalDomain(col("content")))
                .withColumn("text_length", length(col("content")))
                .withColumn("processed_time", current_timestamp())
                .select(
                        col("text_id"),
                        col("ref_id"),
                        col("pmid"),
                        col("article_title").alias("title"),
                        col("journal"),
                        col("year"),
                        col("content"),
                        col("domain"),
                        col("text_length"),
                        col("processed_time")
                )
                .dropDuplicates("pmid");
    }

    // 从引用文本中提取标题（简化版）
    private static Column extractTitleFromCitation(Column content) {
        // 标题通常在年份之后，期刊名之前
        // 这是一个简化的实现，可能需要根据实际情况调整
        return when(col("content").contains("."),
                split(col("content"), "\\.").getItem(1))
                .otherwise(col("content"));
    }

    // 医学领域分类
    private static Column classifyMedicalDomain(Column content) {
        Column lowerContent = lower(content);

        return when(lowerContent.contains("cardiology")
                .or(lowerContent.contains("heart"))
                .or(lowerContent.contains("cardiac"))
                .or(lowerContent.contains("cardiovascular"))
                .or(lowerContent.contains("coronary")), "cardiology")

                .when(lowerContent.contains("neurology")
                        .or(lowerContent.contains("brain"))
                        .or(lowerContent.contains("neural"))
                        .or(lowerContent.contains("neurological"))
                        .or(lowerContent.contains("alzheimer"))
                        .or(lowerContent.contains("parkinson"))
                        .or(lowerContent.contains("demyelinating")), "neurology")

                .when(lowerContent.contains("oncology")
                        .or(lowerContent.contains("cancer"))
                        .or(lowerContent.contains("tumor"))
                        .or(lowerContent.contains("malignant"))
                        .or(lowerContent.contains("chemotherapy"))
                        .or(lowerContent.contains("carcinoma")), "oncology")

                .when(lowerContent.contains("immunology")
                        .or(lowerContent.contains("immune"))
                        .or(lowerContent.contains("antibody"))
                        .or(lowerContent.contains("immunoregulatory"))
                        .or(lowerContent.contains("autoreactive"))
                        .or(lowerContent.contains("vaccination")), "immunology")

                .when(lowerContent.contains("genetics")
                        .or(lowerContent.contains("genomics"))
                        .or(lowerContent.contains("dna"))
                        .or(lowerContent.contains("gene"))
                        .or(lowerContent.contains("mutation")), "genetics")

                .when(lowerContent.contains("biochemistry")
                        .or(lowerContent.contains("molecular"))
                        .or(lowerContent.contains("protein"))
                        .or(lowerContent.contains("enzyme")), "biochemistry")

                .otherwise("general_medicine");
    }

    private static void saveProcessedDataToS3(Dataset<Row> processedData) {
        logger.info("保存数据到S3分区存储...");

        long recordCount = processedData.count();
        int partitions = Math.max(10, Math.min(200, (int)(recordCount / 50000)));
        logger.info("使用 {} 个分区保存数据", partitions);

        Dataset<Row> optimizedData = processedData.repartition(partitions);

        // 关键修改：统一保存路径（与demonstrateQueries中的读取路径一致）
        String parquetSavePath = S3_BUCKET + "/10_01/domain_partitioned/";
        optimizedData.write()
                .mode("overwrite")
                .partitionBy("domain", "year")  // 按领域和年份分区
                .option("compression", "gzip")
                .parquet(parquetSavePath);
        logger.info("Parquet数据已保存到: {}", parquetSavePath);

        // 保存摘要JSON（路径不变）
        String jsonSavePath = S3_BUCKET + "/10_01/summary_json/";
        optimizedData.select("text_id", "pmid", "title", "domain", "year", "journal")
                .write()
                .mode("overwrite")
                .partitionBy("domain")
                .option("compression", "gzip")
                .json(jsonSavePath);
        logger.info("JSON摘要已保存到: {}", jsonSavePath);
    }

    private static void demonstrateQueries(SparkSession spark) {
        logger.info("=== 演示查询功能 ===");

        try {
            Dataset<Row> s3Data = spark.read()
                    .parquet(S3_BUCKET + "/10_01/domain_partitioned/");

            // 1. 基本统计
            logger.info("1. 数据总览:");
            s3Data.groupBy("domain")
                    .agg(
                            count("*").alias("document_count"),
                            avg("text_length").alias("avg_length")
                    )
                    .orderBy(desc("document_count"))
                    .show();

            // 2. 按年份统计
            logger.info("2. 按年份统计:");
            s3Data.filter(col("year").isNotNull())
                    .filter(col("year").notEqual(""))
                    .groupBy("year")
                    .count()
                    .orderBy(desc("year"))
                    .show(20);

            // 3. 各领域示例
            logger.info("3. 各领域文档示例:");
            s3Data.groupBy("domain")
                    .agg(first("title").alias("sample_title"))
                    .show(10, false);

            // 4. 性能测试
            logger.info("4. 查询性能测试:");
            long startTime = System.currentTimeMillis();
            long neurologyCount = s3Data
                    .filter(col("domain").equalTo("neurology"))
                    .count();
            long endTime = System.currentTimeMillis();

            logger.info("神经学文档数量: {}", neurologyCount);
            logger.info("查询耗时: {}ms", (endTime - startTime));

        } catch (Exception e) {
            logger.error("查询演示出错: {}", e.getMessage(), e);
        }
    }
}