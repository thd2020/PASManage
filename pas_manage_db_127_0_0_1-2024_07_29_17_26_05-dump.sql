-- MySQL dump 10.13  Distrib 8.0.37, for Linux (aarch64)
--
-- Host: 127.0.0.1    Database: pas_manage_db
-- ------------------------------------------------------
-- Server version	8.0.37-0ubuntu0.22.04.3

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `department`
--

DROP TABLE IF EXISTS `department`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `department` (
  `department_id` bigint NOT NULL AUTO_INCREMENT,
  `department_name` varchar(100) NOT NULL,
  `phone` varchar(20) DEFAULT NULL,
  `hospital_id` bigint NOT NULL,
  PRIMARY KEY (`department_id`),
  KEY `FKn8lq60po1t7p42oslqbk61wnu` (`hospital_id`),
  CONSTRAINT `FKn8lq60po1t7p42oslqbk61wnu` FOREIGN KEY (`hospital_id`) REFERENCES `hospital` (`hospital_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `department`
--

LOCK TABLES `department` WRITE;
/*!40000 ALTER TABLE `department` DISABLE KEYS */;
/*!40000 ALTER TABLE `department` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `doctor`
--

DROP TABLE IF EXISTS `doctor`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `doctor` (
  `doctor_id` bigint NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `title` varchar(100) DEFAULT NULL,
  `department_id` bigint NOT NULL,
  `user_id` bigint NOT NULL,
  PRIMARY KEY (`doctor_id`),
  UNIQUE KEY `UK3q0j5r6i4e9k3afhypo6uljph` (`user_id`),
  KEY `FK75x47tyyeco3xj4cmlhj8v6ta` (`department_id`),
  CONSTRAINT `FK75x47tyyeco3xj4cmlhj8v6ta` FOREIGN KEY (`department_id`) REFERENCES `department` (`department_id`),
  CONSTRAINT `FK9roto9ydtnjfkixvexq5vxyl5` FOREIGN KEY (`user_id`) REFERENCES `user` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `doctor`
--

LOCK TABLES `doctor` WRITE;
/*!40000 ALTER TABLE `doctor` DISABLE KEYS */;
/*!40000 ALTER TABLE `doctor` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `hospital`
--

DROP TABLE IF EXISTS `hospital`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `hospital` (
  `hospital_id` bigint NOT NULL AUTO_INCREMENT,
  `address` varchar(255) DEFAULT NULL,
  `city` varchar(50) DEFAULT NULL,
  `district` varchar(50) DEFAULT NULL,
  `grade` varchar(50) DEFAULT NULL,
  `latitude` double DEFAULT NULL,
  `longitude` double DEFAULT NULL,
  `name` varchar(100) NOT NULL,
  `phone` varchar(20) DEFAULT NULL,
  `postal_code` varchar(10) DEFAULT NULL,
  `province` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`hospital_id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `hospital`
--

LOCK TABLES `hospital` WRITE;
/*!40000 ALTER TABLE `hospital` DISABLE KEYS */;
INSERT INTO `hospital` VALUES (1,'北京市东城区东单帅府园1号','北京市','东城区','三级甲等',39.90403,116.407526,'北京协和医院','010-69156114','100730','北京市');
/*!40000 ALTER TABLE `hospital` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `image`
--

DROP TABLE IF EXISTS `image`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `image` (
  `image_id` bigint NOT NULL AUTO_INCREMENT,
  `image_name` varchar(255) NOT NULL,
  `image_path` varchar(255) NOT NULL,
  `record_id` varchar(255) DEFAULT NULL,
  `patient_id` bigint NOT NULL,
  PRIMARY KEY (`image_id`),
  KEY `FK3x32v1qt21gf4smdo4s1hiymx` (`record_id`),
  KEY `FKt7mcsayua0q490bv6sr2p13nt` (`patient_id`),
  CONSTRAINT `FK3x32v1qt21gf4smdo4s1hiymx` FOREIGN KEY (`record_id`) REFERENCES `imaging_record` (`record_id`),
  CONSTRAINT `FKt7mcsayua0q490bv6sr2p13nt` FOREIGN KEY (`patient_id`) REFERENCES `patient` (`patient_id`)
) ENGINE=InnoDB AUTO_INCREMENT=65 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `image`
--

LOCK TABLES `image` WRITE;
/*!40000 ALTER TABLE `image` DISABLE KEYS */;
INSERT INTO `image` VALUES (1,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(2,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(3,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(4,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(5,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(6,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(7,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(8,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(9,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(10,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(11,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(12,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(13,'1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','/home/thd2020/pas/1/record123/images/1.2.840.113663.1500.1.386983529.3.10.20220620.140.jpg','record123',1),(14,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(15,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(16,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(17,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(18,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(19,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(20,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(21,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(22,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(23,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(24,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(25,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(26,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(27,'mmexport1716130574783.jpg','/home/thd2020/pas/1/record123/images/mmexport1716130574783.jpg','record123',1),(28,'22.png','/home/thd2020/pas/1/record123/images/22.png','record123',1),(29,'22.png','/home/thd2020/pas/1/record123/images/22.png','record123',1),(30,'22.png','/home/thd2020/pas/1/record123/images/22.png','record123',1),(31,'22.png','/home/thd2020/pas/1/record123/images/22.png','record123',1),(32,'22.png','/home/thd2020/pas/1/record123/images/22.png','record123',1),(33,'25.png','/home/thd2020/pas/1/record123/images/25.png','record123',1),(34,'23.png','/home/thd2020/pas/1/record123/images/23.png','record123',1),(35,'39.png','/home/thd2020/pas/1/record123/images/39.png','record123',1),(36,'[object Object]','/home/thd2020/pas/1/record123/images/[object Object]','record123',1),(37,'[object Object]','/home/thd2020/pas/1/record123/images/[object Object]','record123',1),(38,'30.png','/home/thd2020/pas/1/record123/images/30.png','record123',1),(39,'29.png','/home/thd2020/pas/1/record123/images/29.png','record123',1),(40,'28.png','/home/thd2020/pas/1/record123/images/28.png','record123',1),(41,'38.png','/home/thd2020/pas/1/record123/images/38.png','record123',1),(42,'38.png','/home/thd2020/pas/1/record123/images/38.png','record123',1),(43,'22.png','/home/thd2020/pas/1/record123/images/22.png','record123',1),(44,'25.png','/home/thd2020/pas/1/record123/images/25.png','record123',1),(45,'22.png','/home/thd2020/pas/1/record123/images/22.png','record123',1),(46,'37.png','/home/thd2020/pas/1/record123/images/37.png','record123',1),(47,'25.png','/home/thd2020/pas/1/record123/images/25.png','record123',1),(48,'22.png','/home/thd2020/pas/1/record123/images/22.png','record123',1),(49,'22.png','/home/thd2020/pas/1/record123/images/22.png','record123',1),(50,'22.png','/home/thd2020/pas/1/record123/images/22.png','record123',1),(51,'22.png','/home/thd2020/pas/1/record123/images/22.png','record123',1),(52,'28.png','/home/thd2020/pas/1/record123/images/28.png','record123',1),(53,'29.png','/home/thd2020/pas/1/record123/images/29.png','record123',1),(54,'4.png','/home/thd2020/pas/1/record123/images/4.png','record123',1),(55,'38.png','/home/thd2020/pas/1/record123/images/38.png','record123',1),(56,'38.png','/home/thd2020/pas/1/record123/images/38.png','record123',1),(57,'2.png','/home/thd2020/pas/1/record123/images/2.png','record123',1),(58,'2.png','/home/thd2020/pas/1/record123/images/2.png','record123',1),(59,'30.jpg','/home/thd2020/pas/1/record123/images/30.jpg','record123',1),(60,'30.jpg','/home/thd2020/pas/1/record123/images/30.jpg','record123',1),(61,'30.jpg','/home/thd2020/pas/1/record123/images/30.jpg','record123',1),(62,'30.jpg','/home/thd2020/pas/1/record123/images/30.jpg','record123',1),(63,'微信图片_20240725182351.png;','/home/thd2020/pas/1/record123/images/微信图片_20240725182351.png;','record123',1),(64,'67.png;','/home/thd2020/pas/1/record123/images/67.png;','record123',1);
/*!40000 ALTER TABLE `image` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `imaging_record`
--

DROP TABLE IF EXISTS `imaging_record`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `imaging_record` (
  `record_id` varchar(255) NOT NULL,
  `image_count` int DEFAULT NULL,
  `label_count` int DEFAULT NULL,
  `result_description` tinytext,
  `test_date` datetime(6) NOT NULL,
  `test_type` enum('CT','MRI') NOT NULL,
  `patient_id` bigint NOT NULL,
  PRIMARY KEY (`record_id`),
  KEY `FKfikfhux4xweksylghny2lpn2x` (`patient_id`),
  CONSTRAINT `FKfikfhux4xweksylghny2lpn2x` FOREIGN KEY (`patient_id`) REFERENCES `patient` (`patient_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `imaging_record`
--

LOCK TABLES `imaging_record` WRITE;
/*!40000 ALTER TABLE `imaging_record` DISABLE KEYS */;
INSERT INTO `imaging_record` VALUES ('record123',0,0,'string','2024-07-19 00:49:56.243000','CT',1);
/*!40000 ALTER TABLE `imaging_record` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `mask`
--

DROP TABLE IF EXISTS `mask`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mask` (
  `mask_id` bigint NOT NULL AUTO_INCREMENT,
  `segmentation_json_path` varchar(255) DEFAULT NULL,
  `segmentation_mask_path` varchar(255) NOT NULL,
  `segmentation_source` enum('DOCTOR','MODEL') NOT NULL,
  `image_id` bigint NOT NULL,
  PRIMARY KEY (`mask_id`),
  KEY `FKdkpcymtkl5b78ex39lw10wxhq` (`image_id`),
  CONSTRAINT `FKdkpcymtkl5b78ex39lw10wxhq` FOREIGN KEY (`image_id`) REFERENCES `image` (`image_id`)
) ENGINE=InnoDB AUTO_INCREMENT=46 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `mask`
--

LOCK TABLES `mask` WRITE;
/*!40000 ALTER TABLE `mask` DISABLE KEYS */;
INSERT INTO `mask` VALUES (1,NULL,'/home/thd2020/pas/1/record123/masks/1.2.840.113663.1500.1.386983529.3.10.20220620.140_output.jpg','MODEL',13),(2,NULL,'/home/thd2020/pas/1/record123/masks/mmexport1716130574783_mask.jpg','MODEL',18),(3,NULL,'/home/thd2020/pas/1/record123/masks/mmexport1716130574783_mask.jpg','MODEL',19),(4,NULL,'/home/thd2020/pas/1/record123/masks/mmexport1716130574783_mask.jpg','MODEL',20),(5,NULL,'/home/thd2020/pas/1/record123/masks/mmexport1716130574783_mask.jpg','MODEL',21),(6,NULL,'/home/thd2020/pas/1/record123/masks/mmexport1716130574783_mask.jpg','MODEL',22),(7,NULL,'/home/thd2020/pas/1/record123/masks/mmexport1716130574783_mask.jpg','MODEL',23),(8,NULL,'/home/thd2020/pas/1/record123/masks/mmexport1716130574783_mask.jpg','MODEL',24),(9,NULL,'/home/thd2020/pas/1/record123/masks/mmexport1716130574783_mask.jpg','MODEL',25),(10,NULL,'/home/thd2020/pas/1/record123/masks/mmexport1716130574783_mask.jpg','MODEL',26),(11,NULL,'/home/thd2020/pas/1/record123/masks/mmexport1716130574783_mask.jpg','MODEL',27),(12,NULL,'/home/thd2020/pas/1/record123/masks/22_mask.jpg','MODEL',28),(13,NULL,'/home/thd2020/pas/1/record123/masks/22_mask.jpg','MODEL',29),(14,NULL,'/home/thd2020/pas/1/record123/masks/22_mask.jpg','MODEL',30),(15,NULL,'/home/thd2020/pas/1/record123/masks/22_mask.jpg','MODEL',31),(16,NULL,'/home/thd2020/pas/1/record123/masks/25_mask.jpg','MODEL',33),(17,NULL,'/home/thd2020/pas/1/record123/masks/23_mask.jpg','MODEL',34),(18,NULL,'/home/thd2020/pas/1/record123/masks/39_mask.jpg','MODEL',35),(19,NULL,'/home/thd2020/pas/1/record123/masks/[object Object]_mask.jpg','MODEL',36),(20,NULL,'/home/thd2020/pas/1/record123/masks/[object Object]_mask.jpg','MODEL',37),(21,NULL,'/home/thd2020/pas/1/record123/masks/30_mask.jpg','MODEL',38),(22,NULL,'/home/thd2020/pas/1/record123/masks/29_mask.jpg','MODEL',39),(23,NULL,'/home/thd2020/pas/1/record123/masks/28_mask.jpg','MODEL',40),(24,NULL,'/home/thd2020/pas/1/record123/masks/38_mask.jpg','MODEL',42),(25,NULL,'/home/thd2020/pas/1/record123/masks/38_mask.jpg','MODEL',41),(26,NULL,'/home/thd2020/pas/1/record123/masks/22_mask.jpg','MODEL',43),(27,NULL,'/home/thd2020/pas/1/record123/masks/25_mask.jpg','MODEL',44),(28,NULL,'/home/thd2020/pas/1/record123/masks/22_mask.jpg','MODEL',45),(29,NULL,'/home/thd2020/pas/1/record123/masks/37_mask.jpg','MODEL',46),(30,NULL,'/home/thd2020/pas/1/record123/masks/25_mask.jpg','MODEL',47),(31,NULL,'/home/thd2020/pas/1/record123/masks/22_mask.jpg','MODEL',48),(32,NULL,'/home/thd2020/pas/1/record123/masks/22_mask.jpg','MODEL',49),(33,NULL,'/home/thd2020/pas/1/record123/masks/22_mask.jpg','MODEL',50),(34,NULL,'/home/thd2020/pas/1/record123/masks/22_mask.jpg','MODEL',51),(35,NULL,'/home/thd2020/pas/1/record123/masks/28_mask.jpg','MODEL',52),(36,NULL,'/home/thd2020/pas/1/record123/masks/29_mask.jpg','MODEL',53),(37,NULL,'/home/thd2020/pas/1/record123/masks/4_mask.jpg','MODEL',54),(38,NULL,'/home/thd2020/pas/1/record123/masks/38_mask.jpg','MODEL',55),(39,NULL,'/home/thd2020/pas/1/record123/masks/38_mask.jpg','MODEL',56),(40,NULL,'/home/thd2020/pas/1/record123/masks/2_mask.jpg','MODEL',57),(41,NULL,'/home/thd2020/pas/1/record123/masks/2_mask.jpg','MODEL',58),(42,NULL,'/home/thd2020/pas/1/record123/masks/30_mask.jpg','MODEL',59),(43,NULL,'/home/thd2020/pas/1/record123/masks/30_mask.jpg','MODEL',60),(44,NULL,'/home/thd2020/pas/1/record123/masks/30_mask.jpg','MODEL',61),(45,NULL,'/home/thd2020/pas/1/record123/masks/30_mask.jpg','MODEL',62);
/*!40000 ALTER TABLE `mask` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `medical_record`
--

DROP TABLE IF EXISTS `medical_record`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `medical_record` (
  `record_id` bigint NOT NULL AUTO_INCREMENT,
  `afp` float DEFAULT NULL,
  `age` int DEFAULT NULL,
  `anemia` enum('MILD','MODERATE','NONE') DEFAULT NULL,
  `artery_embolization_history` bit(1) DEFAULT NULL,
  `assisted_reproduction` int DEFAULT NULL,
  `b_hcg` float DEFAULT NULL,
  `c_section_count` int DEFAULT NULL,
  `current_pregnancy_count` int DEFAULT NULL,
  `diabetes` bit(1) DEFAULT NULL,
  `diagnosis` tinytext,
  `fibroid_surgery_history` bit(1) DEFAULT NULL,
  `gravidity` int DEFAULT NULL,
  `height_cm` float DEFAULT NULL,
  `hypertension` bit(1) DEFAULT NULL,
  `inhibina` float DEFAULT NULL,
  `medical_abortion` int DEFAULT NULL,
  `name` varchar(100) DEFAULT NULL,
  `notes` tinytext,
  `pas_history` bit(1) DEFAULT NULL,
  `pre_delivery_weight` float DEFAULT NULL,
  `surgical_abortion` int DEFAULT NULL,
  `symptoms` tinytext,
  `treatment` tinytext,
  `ue3` float DEFAULT NULL,
  `uterine_surgery_history` bit(1) DEFAULT NULL,
  `vaginal_delivery_count` int DEFAULT NULL,
  `visit_date` datetime(6) NOT NULL,
  `visit_type` varchar(100) DEFAULT NULL,
  `doctor_id` bigint DEFAULT NULL,
  `patient_id` bigint NOT NULL,
  PRIMARY KEY (`record_id`),
  KEY `FKmommgymv6rayvbje0hp4c6g8w` (`doctor_id`),
  KEY `FKt0lf3feuiurr73bpln2n6x0v` (`patient_id`),
  CONSTRAINT `FKmommgymv6rayvbje0hp4c6g8w` FOREIGN KEY (`doctor_id`) REFERENCES `doctor` (`doctor_id`),
  CONSTRAINT `FKt0lf3feuiurr73bpln2n6x0v` FOREIGN KEY (`patient_id`) REFERENCES `patient` (`patient_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `medical_record`
--

LOCK TABLES `medical_record` WRITE;
/*!40000 ALTER TABLE `medical_record` DISABLE KEYS */;
/*!40000 ALTER TABLE `medical_record` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `patient`
--

DROP TABLE IF EXISTS `patient`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `patient` (
  `patient_id` bigint NOT NULL AUTO_INCREMENT,
  `address` varchar(255) DEFAULT NULL,
  `birth_date` date DEFAULT NULL,
  `gender` enum('FEMALE','MALE','OTHER') DEFAULT NULL,
  `name` varchar(100) NOT NULL,
  `pass_id` varchar(20) DEFAULT NULL,
  `doctor_id` bigint DEFAULT NULL,
  `user_id` bigint DEFAULT NULL,
  PRIMARY KEY (`patient_id`),
  UNIQUE KEY `UK6i3fp8wcdxk473941mbcvdao4` (`user_id`),
  KEY `FKmer5utvy1hiff7ovs6f4bjtnw` (`doctor_id`),
  CONSTRAINT `FKmer5utvy1hiff7ovs6f4bjtnw` FOREIGN KEY (`doctor_id`) REFERENCES `doctor` (`doctor_id`),
  CONSTRAINT `FKp6ttmfrxo2ejiunew4ov805uc` FOREIGN KEY (`user_id`) REFERENCES `user` (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `patient`
--

LOCK TABLES `patient` WRITE;
/*!40000 ALTER TABLE `patient` DISABLE KEYS */;
INSERT INTO `patient` VALUES (1,NULL,NULL,NULL,'李子阳','123456789012345678',NULL,1),(4,'北京市朝阳区某街道','1980-01-01','MALE','反而非','510106200001011000',NULL,4),(6,NULL,NULL,NULL,'逆天',NULL,NULL,8),(7,NULL,NULL,NULL,'aaa',NULL,NULL,9),(10,'北京市朝阳区某街道','1980-01-01','MALE','里斯','110106200001001000',NULL,2),(14,'北京市朝阳区某街道','1980-01-01','MALE','dfhdfh','123456789012345678',NULL,10),(15,'北京市朝阳区某街道','1980-01-01','MALE','dfhdfh','123456789012345679',NULL,11),(16,'北京市朝阳区某街道','1980-01-01','MALE','张三','123456789012355678',NULL,12);
/*!40000 ALTER TABLE `patient` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `placenta_segmentation_grading`
--

DROP TABLE IF EXISTS `placenta_segmentation_grading`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `placenta_segmentation_grading` (
  `seg_grade_id` bigint NOT NULL AUTO_INCREMENT,
  `grade` enum('ADHESION','INVASION','NORMAL','PENETRATION') DEFAULT NULL,
  `overall_grade` enum('ADHESION','INVASION','NORMAL','PENETRATION') DEFAULT NULL,
  `probability` float DEFAULT NULL,
  `segmentation_source` enum('DOCTOR','MODEL') NOT NULL,
  `timestamp` datetime(6) NOT NULL,
  `image_id` bigint NOT NULL,
  `mask_id` bigint NOT NULL,
  `patient_id` bigint NOT NULL,
  PRIMARY KEY (`seg_grade_id`),
  KEY `FK63tfijp5eb4lvst11vetg6qwm` (`image_id`),
  KEY `FKe3dn69u60gmwg0acrvand8il2` (`mask_id`),
  KEY `FKl4i428apxmrvxv1jhxsxpaqib` (`patient_id`),
  CONSTRAINT `FK63tfijp5eb4lvst11vetg6qwm` FOREIGN KEY (`image_id`) REFERENCES `image` (`image_id`),
  CONSTRAINT `FKe3dn69u60gmwg0acrvand8il2` FOREIGN KEY (`mask_id`) REFERENCES `mask` (`mask_id`),
  CONSTRAINT `FKl4i428apxmrvxv1jhxsxpaqib` FOREIGN KEY (`patient_id`) REFERENCES `patient` (`patient_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `placenta_segmentation_grading`
--

LOCK TABLES `placenta_segmentation_grading` WRITE;
/*!40000 ALTER TABLE `placenta_segmentation_grading` DISABLE KEYS */;
/*!40000 ALTER TABLE `placenta_segmentation_grading` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `surgery_and_blood_test`
--

DROP TABLE IF EXISTS `surgery_and_blood_test`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `surgery_and_blood_test` (
  `record_id` bigint NOT NULL AUTO_INCREMENT,
  `aortic_balloon` bit(1) DEFAULT NULL,
  `assisting_surgeon` varchar(100) DEFAULT NULL,
  `gestational_weeks` int DEFAULT NULL,
  `hospital_stay_days` int DEFAULT NULL,
  `hysterectomy` bit(1) DEFAULT NULL,
  `intraoperative_bleeding` int DEFAULT NULL,
  `notes` tinytext,
  `plasma_transfused` int DEFAULT NULL,
  `postoperative24h_hb` float DEFAULT NULL,
  `postoperative24h_hct` float DEFAULT NULL,
  `postoperative_transfusion_status` varchar(100) DEFAULT NULL,
  `pre_delivery_bleeding` int DEFAULT NULL,
  `preoperative_hb` float DEFAULT NULL,
  `preoperative_hct` float DEFAULT NULL,
  `primary_surgeon` varchar(100) DEFAULT NULL,
  `red_blood_cells_transfused` int DEFAULT NULL,
  `patient_id` bigint NOT NULL,
  PRIMARY KEY (`record_id`),
  KEY `FKqekseiydqelutsu2ms9kqaoet` (`patient_id`),
  CONSTRAINT `FKqekseiydqelutsu2ms9kqaoet` FOREIGN KEY (`patient_id`) REFERENCES `patient` (`patient_id`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `surgery_and_blood_test`
--

LOCK TABLES `surgery_and_blood_test` WRITE;
/*!40000 ALTER TABLE `surgery_and_blood_test` DISABLE KEYS */;
INSERT INTO `surgery_and_blood_test` VALUES (5,_binary '','李医生',39,5,_binary '\0',1000,'手术顺利，患者恢复良好',3,110.5,32.5,'无',500,120.5,35,'张医生',2,1),(6,_binary '','李医生',39,5,_binary '\0',1000,'手术顺利，患者恢复良好',3,110.5,32.5,'无',500,120.5,35,'张医生',2,1),(7,_binary '','李医生',39,5,_binary '\0',1000,'手术顺利，患者恢复良好',3,110.5,32.5,'无',500,120.5,35,'张医生',2,1),(8,_binary '','李医生',39,5,_binary '\0',1000,'手术顺利，患者恢复良好',3,110.5,32.5,'无',500,120.5,35,'张医生',2,1),(9,_binary '','李医生',39,5,_binary '\0',1000,'手术顺利，患者恢复良好',3,110.5,32.5,'无',500,120.5,35,'张医生',2,1),(10,_binary '','李医生',39,5,_binary '\0',1000,'手术顺利，患者恢复良好',3,110.5,32.5,'无',500,120.5,35,'张医生',2,1);
/*!40000 ALTER TABLE `surgery_and_blood_test` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ultrasound_score`
--

DROP TABLE IF EXISTS `ultrasound_score`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ultrasound_score` (
  `score_id` bigint NOT NULL AUTO_INCREMENT,
  `blood_flow_signals` int DEFAULT NULL,
  `cervical_shape` int DEFAULT NULL,
  `estimated_blood_loss` int DEFAULT NULL,
  `examination_date` datetime(6) NOT NULL,
  `fetal_position` int DEFAULT NULL,
  `myometrium_invasion` int DEFAULT NULL,
  `placental_body_position` int DEFAULT NULL,
  `placental_lacunae` int DEFAULT NULL,
  `placental_position` int DEFAULT NULL,
  `placental_thickness` int DEFAULT NULL,
  `record_id` varchar(255) DEFAULT NULL,
  `suspected_invasion_range` int DEFAULT NULL,
  `suspected_placenta_location` int DEFAULT NULL,
  `total_score` int DEFAULT NULL,
  `patient_id` bigint NOT NULL,
  PRIMARY KEY (`score_id`),
  KEY `FK3ldsyw6f12qp4ip7syntbdro` (`patient_id`),
  CONSTRAINT `FK3ldsyw6f12qp4ip7syntbdro` FOREIGN KEY (`patient_id`) REFERENCES `patient` (`patient_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ultrasound_score`
--

LOCK TABLES `ultrasound_score` WRITE;
/*!40000 ALTER TABLE `ultrasound_score` DISABLE KEYS */;
/*!40000 ALTER TABLE `ultrasound_score` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `user`
--

DROP TABLE IF EXISTS `user`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user` (
  `user_id` bigint NOT NULL AUTO_INCREMENT,
  `created_at` datetime(6) NOT NULL,
  `email` varchar(100) DEFAULT NULL,
  `last_login` datetime(6) DEFAULT NULL,
  `password` varchar(255) NOT NULL,
  `phone` varchar(20) DEFAULT NULL,
  `role` enum('ADMIN','B_DOCTOR','PATIENT','T_DOCTOR') NOT NULL,
  `status` enum('ACTIVE','BANNED','INACTIVE') NOT NULL,
  `username` varchar(50) NOT NULL,
  `provider` enum('GOOGLE','LOCAL') DEFAULT NULL,
  `doctor_id` bigint DEFAULT NULL,
  `hospital_id` bigint DEFAULT NULL,
  `patient_id` bigint DEFAULT NULL,
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `UK1ttoi0gjnhju8ecfur1en8cdu` (`doctor_id`),
  UNIQUE KEY `UKq50qlaoc211y22tfcg3mom7a4` (`hospital_id`),
  UNIQUE KEY `UKekvwufxfefmd2gadw28u4uncv` (`patient_id`),
  CONSTRAINT `FK50ye93xm2iw5ttgidh543nryv` FOREIGN KEY (`patient_id`) REFERENCES `patient` (`patient_id`),
  CONSTRAINT `FKmo2mvccns3uvgpuk7vgrtcjq9` FOREIGN KEY (`hospital_id`) REFERENCES `hospital` (`hospital_id`),
  CONSTRAINT `FKn394rrbe4yseqesqx870tlstj` FOREIGN KEY (`doctor_id`) REFERENCES `doctor` (`doctor_id`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user`
--

LOCK TABLES `user` WRITE;
/*!40000 ALTER TABLE `user` DISABLE KEYS */;
INSERT INTO `user` VALUES (1,'2024-06-25 10:06:52.432457','john.doe@example.com',NULL,'$2a$10$GW9/QbnLrqMdzrcm3NAaTe5oBsXEe/xXUhtiDNMMnNieV9xae/yRS','13981763058','PATIENT','ACTIVE','赵子龙',NULL,NULL,NULL,NULL),(2,'2024-06-25 10:13:50.616522','thd2020@gmail.com','2024-07-26 08:57:48.695246','$2a$10$sEsDCSTuaZh4WCvdoXALMOrl05L234mH6.rRyk74LXAukLq7qceq6','1234567890','ADMIN','ACTIVE','thd2020',NULL,NULL,NULL,NULL),(3,'2024-06-28 03:09:43.208533','test@test.com',NULL,'$2a$10$s14m6XdXTszecQqWQAhInuhFLWUNJOtweOBJOGC2Br78hB/6HEASi','110','B_DOCTOR','ACTIVE','justAJoke',NULL,NULL,NULL,NULL),(4,'2024-07-05 09:08:12.817028','thdfreelancer@gmail.com',NULL,'$2a$10$kRz0rcXU8xoKmjol/jiBi.N2FCS2ZmIEKSL5i0DlHqtFdXMCL7g2u',NULL,'PATIENT','ACTIVE','free thd','GOOGLE',NULL,NULL,NULL),(8,'2024-07-24 09:54:08.627836',NULL,NULL,'$2a$10$CNrD/wPaR89Zj4NkcTUr3..7ZMVKF2MF6gLOBPPQMHeF00UmAkFay','52013141314','PATIENT','ACTIVE','逆天',NULL,NULL,NULL,NULL),(9,'2024-07-24 09:56:57.579348',NULL,NULL,'$2a$10$Bk/GZFRteUiRranYwEy94.d.tTkwxamoEu1.exIPzthFMcfTbo852','52013141314','PATIENT','ACTIVE','aaa',NULL,NULL,1,NULL),(10,'2024-07-26 07:25:40.708311',NULL,NULL,'$2a$10$NMw5ebpmF8/3wD50/PLaEe0q0TeYDLxrqsS0emQlzECM7qKrqIL.q',NULL,'PATIENT','ACTIVE','dfhdfh',NULL,NULL,NULL,NULL),(11,'2024-07-26 07:26:46.716529',NULL,NULL,'$2a$10$6OxySUmKNOWKNqXR33lQr.eEufvIjUSPXqzPsUbuHZMsYn5vjzyAi',NULL,'PATIENT','ACTIVE','dfhdfh',NULL,NULL,NULL,NULL),(12,'2024-07-26 07:34:39.142251',NULL,NULL,'$2a$10$ySOnq4ex3Kw3uUpvbtURBuVcN8cyBAGiLVSVAS6R0BmuEwO/No.OC',NULL,'PATIENT','ACTIVE','张三',NULL,NULL,NULL,NULL);
/*!40000 ALTER TABLE `user` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-07-29 17:26:23
