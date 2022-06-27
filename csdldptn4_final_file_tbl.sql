-- MySQL dump 10.13  Distrib 8.0.28, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: csdldptn4_final
-- ------------------------------------------------------
-- Server version	8.0.28

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `file_tbl`
--

DROP TABLE IF EXISTS `file_tbl`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `file_tbl` (
  `ID` int NOT NULL,
  `filename` varchar(255) DEFAULT NULL,
  `filepath` varchar(255) DEFAULT NULL,
  `label` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `file_tbl`
--

LOCK TABLES `file_tbl` WRITE;
/*!40000 ALTER TABLE `file_tbl` DISABLE KEYS */;
INSERT INTO `file_tbl` VALUES (1,'1 (1)','./static/data/1 (1)','Mau van tay 1'),(2,'1 (2)','./static/data/1 (2)','Mau van tay 1'),(3,'1 (3)','./static/data/1 (3)','Mau van tay 1'),(4,'1 (4)','./static/data/1 (4)','Mau van tay 1'),(5,'1 (5)','./static/data/1 (5)','Mau van tay 1'),(6,'2 (1)','./static/data/2 (1)','Mau van tay 2'),(7,'2 (2)','./static/data/2 (2)','Mau van tay 2'),(8,'2 (3)','./static/data/2 (3)','Mau van tay 2'),(9,'2 (4)','./static/data/2 (4)','Mau van tay 2'),(10,'2 (5)','./static/data/2 (5)','Mau van tay 2'),(11,'3 (1)','./static/data/3 (1)','Mau van tay 3'),(12,'3 (2)','./static/data/3 (2)','Mau van tay 3'),(13,'3 (3)','./static/data/3 (3)','Mau van tay 3'),(14,'3 (4)','./static/data/3 (4)','Mau van tay 3'),(15,'3 (5)','./static/data/3 (5)','Mau van tay 3'),(16,'4 (1)','./static/data/4 (1)','Mau van tay 4'),(17,'4 (2)','./static/data/4 (2)','Mau van tay 4'),(18,'4 (3)','./static/data/4 (3)','Mau van tay 4'),(19,'4 (4)','./static/data/4 (4)','Mau van tay 4'),(20,'4 (5)','./static/data/4 (5)','Mau van tay 4'),(21,'5 (1)','./static/data/5 (1)','Mau van tay 5'),(22,'5 (2)','./static/data/5 (2)','Mau van tay 5'),(23,'5 (3)','./static/data/5 (3)','Mau van tay 5'),(24,'5 (4)','./static/data/5 (4)','Mau van tay 5'),(25,'5 (5)','./static/data/5 (5)','Mau van tay 5'),(26,'6 (1)','./static/data/6 (1)','Mau van tay 6'),(27,'6 (2)','./static/data/6 (2)','Mau van tay 6'),(28,'6 (3)','./static/data/6 (3)','Mau van tay 6'),(29,'6 (4)','./static/data/6 (4)','Mau van tay 6'),(30,'6 (5)','./static/data/6 (5)','Mau van tay 6'),(31,'7 (1)','./static/data/7 (1)','Mau van tay 7'),(32,'7 (2)','./static/data/7 (2)','Mau van tay 7'),(33,'7 (3)','./static/data/7 (3)','Mau van tay 7'),(34,'7 (4)','./static/data/7 (4)','Mau van tay 7'),(35,'7 (5)','./static/data/7 (5)','Mau van tay 7'),(36,'8 (1)','./static/data/8 (1)','Mau van tay 8'),(37,'8 (2)','./static/data/8 (2)','Mau van tay 8'),(38,'8 (3)','./static/data/8 (3)','Mau van tay 8'),(39,'8 (4)','./static/data/8 (4)','Mau van tay 8'),(40,'8 (5)','./static/data/8 (5)','Mau van tay 8'),(41,'9 (1)','./static/data/9 (1)','Mau van tay 9'),(42,'9 (2)','./static/data/9 (2)','Mau van tay 9'),(43,'9 (3)','./static/data/9 (3)','Mau van tay 9'),(44,'9 (4)','./static/data/9 (4)','Mau van tay 9'),(45,'9 (5)','./static/data/9 (5)','Mau van tay 9'),(46,'10 (1)','./static/data/10 (1)','Mau van tay 10'),(47,'10 (2)','./static/data/10 (2)','Mau van tay 10'),(48,'10 (3)','./static/data/10 (3)','Mau van tay 10'),(49,'10 (4)','./static/data/10 (4)','Mau van tay 10'),(50,'10 (5)','./static/data/10 (5)','Mau van tay 10'),(51,'11 (1)','./static/data/11 (1)','Mau van tay 11'),(52,'11 (2)','./static/data/11 (2)','Mau van tay 11'),(53,'11 (3)','./static/data/11 (3)','Mau van tay 11'),(54,'11 (4)','./static/data/11 (4)','Mau van tay 11'),(55,'11 (5)','./static/data/11 (5)','Mau van tay 11'),(56,'12 (1)','./static/data/12 (1)','Mau van tay 12'),(57,'12 (2)','./static/data/12 (2)','Mau van tay 12'),(58,'12 (3)','./static/data/12 (3)','Mau van tay 12'),(59,'12 (4)','./static/data/12 (4)','Mau van tay 12'),(60,'12 (5)','./static/data/12 (5)','Mau van tay 12'),(61,'13 (1)','./static/data/13 (1)','Mau van tay 13'),(62,'13 (2)','./static/data/13 (2)','Mau van tay 13'),(63,'13 (3)','./static/data/13 (3)','Mau van tay 13'),(64,'13 (4)','./static/data/13 (4)','Mau van tay 13'),(65,'13 (5)','./static/data/13 (5)','Mau van tay 13'),(66,'14 (1)','./static/data/14 (1)','Mau van tay 14'),(67,'14 (2)','./static/data/14 (2)','Mau van tay 14'),(68,'14 (3)','./static/data/14 (3)','Mau van tay 14'),(69,'14 (4)','./static/data/14 (4)','Mau van tay 14'),(70,'14 (5)','./static/data/14 (5)','Mau van tay 14'),(71,'15 (1)','./static/data/15 (1)','Mau van tay 15'),(72,'15 (2)','./static/data/15 (2)','Mau van tay 15'),(73,'15 (3)','./static/data/15 (3)','Mau van tay 15'),(74,'15 (4)','./static/data/15 (4)','Mau van tay 15'),(75,'15 (5)','./static/data/15 (5)','Mau van tay 15'),(76,'16 (1)','./static/data/16 (1)','Mau van tay 16'),(77,'16 (2)','./static/data/16 (2)','Mau van tay 16'),(78,'16 (3)','./static/data/16 (3)','Mau van tay 16'),(79,'16 (4)','./static/data/16 (4)','Mau van tay 16'),(80,'16 (5)','./static/data/16 (5)','Mau van tay 16'),(81,'17 (1)','./static/data/17 (1)','Mau van tay 17'),(82,'17 (2)','./static/data/17 (2)','Mau van tay 17'),(83,'17 (3)','./static/data/17 (3)','Mau van tay 17'),(84,'17 (4)','./static/data/17 (4)','Mau van tay 17'),(85,'17 (5)','./static/data/17 (5)','Mau van tay 17'),(86,'18 (1)','./static/data/18 (1)','Mau van tay 18'),(87,'18 (2)','./static/data/18 (2)','Mau van tay 18'),(88,'18 (3)','./static/data/18 (3)','Mau van tay 18'),(89,'18 (4)','./static/data/18 (4)','Mau van tay 18'),(90,'18 (5)','./static/data/18 (5)','Mau van tay 18'),(91,'19 (1)','./static/data/19 (1)','Mau van tay 19'),(92,'19 (2)','./static/data/19 (2)','Mau van tay 19'),(93,'19 (3)','./static/data/19 (3)','Mau van tay 19'),(94,'19 (4)','./static/data/19 (4)','Mau van tay 19'),(95,'19 (5)','./static/data/19 (5)','Mau van tay 19'),(96,'20 (1)','./static/data/20 (1)','Mau van tay 20'),(97,'20 (2)','./static/data/20 (2)','Mau van tay 20'),(98,'20 (3)','./static/data/20 (3)','Mau van tay 20'),(99,'20 (4)','./static/data/20 (4)','Mau van tay 20'),(100,'20 (5)','./static/data/20 (5)','Mau van tay 20');
/*!40000 ALTER TABLE `file_tbl` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2022-06-21  0:28:14
