.mode csv

DROP TABLE IF EXISTS "biodiversity";
DROP TABLE IF EXISTS "pollution";
DROP TABLE IF EXISTS "carbon_dioxide";

CREATE TABLE IF NOT EXISTS "carbon_dioxide"(
"Year" TIMESTAMP, "Annual" NUMBER, "Annual - Global (parts per million)" NUMBER, "January (parts per million)" NUMBER,
 "February (parts per million)" NUMBER, "March (parts per million)" NUMBER, "April (parts per million)" NUMBER, "May (parts per million)" NUMBER,
 "June (parts per million)" NUMBER, "July (parts per million)" NUMBER, "August (parts per million)" NUMBER, "September (parts per million)" NUMBER,
 "October (parts per million)" NUMBER, "November (parts per million)" NUMBER, "December (parts per million)" NUMBER);

CREATE TABLE "biodiversity"(
"Year" TIMESTAMP, "national index" NUMBER, "number of species" NUMBER, "birds index" NUMBER,
 "number of bird species" NUMBER, "mammals index" NUMBER, "number of mammal species" NUMBER, "fish index" NUMBER,
 "number of fish species" NUMBER);


CREATE TABLE "pollution"(
"Year" TIMESTAMP, "Sulphur oxides" NUMBER, "Nitrogen oxides" NUMBER, "Volatile organic compounds" NUMBER,
 "Ammonia" NUMBER, "Carbon monoxide" NUMBER, "Fine particulate matter" NUMBER);


.import datasets/biodiversity.csv biodiversity
.import datasets/pollution.csv pollution
.import datasets/carbon-dioxide.csv carbon_dioxide
