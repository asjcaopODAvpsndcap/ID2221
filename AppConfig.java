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
                System.err.println("Error: Configuration file 'config.properties' not found!");
            }
            props.load(input);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    // S3 Path Configurations
    public static final String S3_BUCKET = props.getProperty("s3.bucket", "s3a://id2221");
    public static final String RAW_XML_PATH = S3_BUCKET + "/medical_xml_data/";
    public static final String PROCESSED_PARQUET_PATH = S3_BUCKET + "/processed_data/domain_partitioned/";
    public static final String PROCESSED_JSON_PATH = S3_BUCKET + "/processed_data/summary_json/";

    // Default Memory Configurations
    public static final String DEFAULT_DRIVER_MEMORY = "4g";
    public static final String DEFAULT_EXECUTOR_MEMORY = "4g";

    // Medical Domain Keywords (key: domain label, value: list of keywords)
    public static final Map<String, List<String>> DOMAIN_KEYWORDS = new HashMap<>() {{
        put("cardiology", Arrays.asList("cardiology", "heart", "cardiac", "cardiovascular", "coronary"));
        put("neurology", Arrays.asList("neurology", "brain", "neural", "neurological", "alzheimer", "parkinson"));
        put("general_medicine", Collections.emptyList()); // Default domain
    }};

    // AWS Access Key Configurations (read from properties file)
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