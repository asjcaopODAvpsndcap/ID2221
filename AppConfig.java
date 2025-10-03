package org.heima;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

public class AppConfig {
    private static final Properties props = new Properties();

    static {
        try (InputStream input = AppConfig.class.getClassLoader().getResourceAsStream("config.properties")) {
            if (input == null) {
                System.err.println("配置文件 config.properties 未找到！");

            }
            props.load(input);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    // S3 路径相关配置
    public static final String S3_BUCKET = props.getProperty("s3.bucket", "s3a://id2221");
    public static final String RAW_XML_PATH = S3_BUCKET + "/medical_xml_data/";
    public static final String PROCESSED_PARQUET_PATH = S3_BUCKET + "/processed_data/domain_partitioned/";
    public static final String PROCESSED_JSON_PATH = S3_BUCKET + "/processed_data/summary_json/";

    // 内存配置默认值
    public static final String DEFAULT_DRIVER_MEMORY = "4g";
    public static final String DEFAULT_EXECUTOR_MEMORY = "4g";

    // 医学领域关键词（key：领域标签，value：关键词列表）
    public static final Map<String, List<String>> DOMAIN_KEYWORDS = new HashMap<>() {{
        put("cardiology", Arrays.asList("cardiology", "heart", "cardiac", "cardiovascular", "coronary"));
        put("neurology", Arrays.asList("neurology", "brain", "neural", "neurological", "alzheimer", "parkinson"));
        put("general_medicine", Collections.emptyList()); // 默认领域
    }};

    // AWS 访问密钥相关配置，从配置文件读取
    public static String getAwsAccessKeyId() {
        return props.getProperty("aws.accessKeyId");
    }

    public static String getAwsSecretAccessKey() {
        return props.getProperty("aws.secretAccessKey");
    }

    public static String getS3Endpoint() {
        return props.getProperty("s3.endpoint", "s3.eu-north-1.amazonaws.com");
    }
}