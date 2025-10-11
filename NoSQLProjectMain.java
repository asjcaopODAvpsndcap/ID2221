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
            logger.info("Starting to process PubMed XML data...");

            // 1. Load data
            Dataset<Row> rawData = loadPubMedData(spark);
            logger.info("=== Raw Data Check ===");
            logger.info("Number of raw data entries: {}", rawData.count());
            rawData.printSchema();
            rawData.show(10, false);

            // 2. Preprocess data
            Dataset<Row> processedData = preprocessData(spark, rawData);
            logger.info("=== Processed Data Check ===");
            logger.info("Number of processed data entries: {}", processedData.count());

            if (processedData.count() == 0) {
                logger.error("⚠️ Warning: No data after processing!");
                return;
            }

            // 3. Display sample data
            logger.info("Data sample:");
            processedData.show(10, false);

            // 4. Statistics by domain
            logger.info("Classification by medical domain:");
            processedData.groupBy("domain").count()
                    .orderBy(desc("count"))
                    .show();

            // 5. Save processed data to S3
            logger.info("Saving data to S3...");
            saveProcessedDataToS3(processedData);

            // 6. Demonstrate query functionality
            logger.info("Demonstrating query functionality...");
            demonstrateQueries(spark);

        } catch (Exception e) {
            logger.error("An error occurred during processing: {}", e.getMessage(), e);
        } finally {
            spark.stop();
        }
    }

    public static Dataset<Row> loadPubMedData(SparkSession spark) {
        logger.info("Loading full JATS format article XML (root tag: <article>)...");

        // 1. First, explore 5 records to verify the structure (to avoid errors from loading everything at once)
        Dataset<Row> exploratoryData = spark.read()
                .format("com.databricks.spark.xml")
                .option("rootTag", "article")  // ✅ Core fix: The root tag is <article> (not PubmedArticleSet)
                .option("rowTag", "article")   // ✅ The row tag is also <article> (one article per row)
                .option("inferSchema", "true")
                .option("treatEmptyValuesAsNulls", "true")
                .option("attributePrefix", "_")  // To read tag attributes (like ref's id)
                .option("ignoreSurroundingSpaces", "true")
                .load(S3_BUCKET + "/medical_xml_data/")  // Your S3 path
                .limit(5);

        // Print Schema and data to confirm fields exist (critical validation step)
        logger.info("=== Exploratory Data Schema (confirming front/body/back fields exist) ===");
        exploratoryData.printSchema();
        logger.info("=== Exploratory Data Sample (confirming abstract/title fields are not null) ===");
        exploratoryData.select(
                "front.article-meta.article-id._VALUE",  // PMID
                "front.article-meta.title-group.article-title",  // Title
                "front.article-meta.abstract.p"  // Abstract
        ).show(5, false);

        // 2. Load all data (reusing the correct root/row tags)
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
        logger.info("Starting data preprocessing (based on actual XML fields)...");

        long initialCount = rawData.count();
        logger.info("Initial data count: {}", initialCount);
        if (initialCount == 0) {
            logger.error("❌ No data after loading! Please check the XML path and rootTag.");
            return rawData;
        }

        // ✅ Core fix: Instead of accessing deep paths at once, extract step-by-step safely.
        // We will extract all needed structures to the top level first, setting them to null if they don't exist.
        Dataset<Row> step1 = rawData
                // Safely extract article-meta and journal-meta structs
                .withColumn("article_meta_struct", when(col("front").isNotNull(), col("front.article-meta")).otherwise(lit(null)))
                .withColumn("journal_meta_struct", when(col("front").isNotNull(), col("front.journal-meta")).otherwise(lit(null)));

        // Now, perform the next extractions based on these safe top-level structs
        Dataset<Row> step2 = step1
                // 1. Extract PMID (from article_meta_struct)
                .withColumn("pmid",
                        when(col("article_meta_struct.article-id").isNotNull()
                                        .and(size(col("article_meta_struct.article-id")).gt(0)),
                                expr("filter(article_meta_struct.`article-id`, x -> x.`_pub-id-type` == 'pmid')[0].`_VALUE`")
                        ).otherwise(lit("unknown")))

                // 2. Extract article title (from article_meta_struct)
                .withColumn("article_title",
                        when(col("article_meta_struct.title-group.article-title").isNotNull(),
                                col("article_meta_struct.title-group.article-title").cast("string"))
                                .otherwise(lit("unknown")))

                // 3. Extract journal name (from journal_meta_struct)
                .withColumn("journal",
                        when(col("journal_meta_struct.journal-title-group.journal-title").isNotNull(),
                                col("journal_meta_struct.journal-title-group.journal-title").cast("string"))
                                .otherwise(lit("unknown")))

                // 4. Extract publication year (from article_meta_struct)
                .withColumn("year",
                        when(col("article_meta_struct.pub-date").isNotNull(),
                                expr("filter(article_meta_struct.`pub-date`, x -> x.`_pub-type` == 'ppub')[0].year")
                        ).otherwise(lit("unknown")))

                // 5. Extract content (from article_meta_struct and body)
                .withColumn("abstract_text",
                        when(col("article_meta_struct.abstract.p").isNotNull(),
                                // First use flatten to reduce ARRAY<ARRAY<STRING>> to ARRAY<STRING>
                                // Then use concat_ws to join
                                concat_ws(" ", flatten(col("article_meta_struct.abstract.p"))))
                                .otherwise(lit("")))


                .withColumn("body_text",
                        when(col("body").isNotNull(),
                                col("body").cast("string"))
                                .otherwise(lit("")))
                .withColumn("citation_text",
                        concat_ws(" ", col("abstract_text"), col("body_text")))

                // 6. Generate a unique reference ID
                .withColumn("ref_id",
                        when(input_file_name().isNotNull(), input_file_name())
                                .otherwise(monotonically_increasing_id().cast("string")));

        // Print extraction results
        logger.info("=== Field Extraction Sample ===");
        step2.select("pmid", "article_title", "journal", "year", "citation_text")
                .show(5, false);

        // Content filtering
        Dataset<Row> step3 = step2
                .filter(col("citation_text").isNotNull().and(length(trim(col("citation_text"))).gt(50)))
                .filter(col("pmid").isNotNull().and(col("pmid").notEqual("unknown")));

        long afterContentFilterCount = step3.count();
        logger.info("Data count after content filtering: {} (filtered out due to inadequate content: {})",
                afterContentFilterCount, initialCount - afterContentFilterCount);

        if (afterContentFilterCount == 0) {
            logger.error("⚠️ No data after content filtering! All records might have empty or too short abstracts or bodies.");
            return step2.select( // Return data that has not been content-filtered, but only select the required fields
                    col("pmid"), col("article_title").alias("title"), col("journal"),
                    col("year"), col("citation_text").alias("content"), col("ref_id")
                    // ... add other necessary fields ...
            );
        }

        // Final data cleaning and selection
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

    // Extract title from citation text (simplified version)
    private static Column extractTitleFromCitation(Column content) {
        // The title is usually after the year and before the journal name
        // This is a simplified implementation and may need adjustment based on the actual situation
        return when(col("content").contains("."),
                split(col("content"), "\\.").getItem(1))
                .otherwise(col("content"));
    }

    // Classify medical domain
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
        logger.info("Saving data to S3 partitioned storage...");

        long recordCount = processedData.count();
        int partitions = Math.max(10, Math.min(200, (int)(recordCount / 50000)));
        logger.info("Using {} partitions to save data", partitions);

        Dataset<Row> optimizedData = processedData.repartition(partitions);

        // Key change: unify the save path (consistent with the read path in demonstrateQueries)
        String parquetSavePath = S3_BUCKET + "/10_01/domain_partitioned/";
        optimizedData.write()
                .mode("overwrite")
                .partitionBy("domain", "year")  // Partition by domain and year
                .option("compression", "gzip")
                .parquet(parquetSavePath);
        logger.info("Parquet data has been saved to: {}", parquetSavePath);

        // Save summary JSON (path unchanged)
        String jsonSavePath = S3_BUCKET + "/10_01/summary_json/";
        optimizedData.select("text_id", "pmid", "title", "domain", "year", "journal")
                .write()
                .mode("overwrite")
                .partitionBy("domain")
                .option("compression", "gzip")
                .json(jsonSavePath);
        logger.info("JSON summary has been saved to: {}", jsonSavePath);
    }

    private static void demonstrateQueries(SparkSession spark) {
        logger.info("=== Demonstrating Query Functionality ===");

        try {
            Dataset<Row> s3Data = spark.read()
                    .parquet(S3_BUCKET + "/10_01/domain_partitioned/");

            // 1. Basic statistics
            logger.info("1. Data overview:");
            s3Data.groupBy("domain")
                    .agg(
                            count("*").alias("document_count"),
                            avg("text_length").alias("avg_length")
                    )
                    .orderBy(desc("document_count"))
                    .show();

            // 2. Statistics by year
            logger.info("2. Statistics by year:");
            s3Data.filter(col("year").isNotNull())
                    .filter(col("year").notEqual(""))
                    .groupBy("year")
                    .count()
                    .orderBy(desc("year"))
                    .show(20);

            // 3. Examples from each domain
            logger.info("3. Document examples from each domain:");
            s3Data.groupBy("domain")
                    .agg(first("title").alias("sample_title"))
                    .show(10, false);

            // 4. Performance test
            logger.info("4. Query performance test:");
            long startTime = System.currentTimeMillis();
            long neurologyCount = s3Data
                    .filter(col("domain").equalTo("neurology"))
                    .count();
            long endTime = System.currentTimeMillis();

            logger.info("Number of neurology documents: {}", neurologyCount);
            logger.info("Query execution time: {}ms", (endTime - startTime));

        } catch (Exception e) {
            logger.error("Error during query demonstration: {}", e.getMessage(), e);
        }
    }
}