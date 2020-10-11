import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{avg, callUDF, col, corr, desc, lit, row_number, udf, variance}

//data https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data?select=AB_NYC_2019.csv

val spark = SparkSession
  .builder()
  .appName("Task_1")
  .config("spark.master", "local")
  .getOrCreate()

val data = spark
  .read
  .option("header", "true")
  .option("mode", "DROPMALFORMED")
  .option("escape", "\"")
  .csv("IdeaProjects/university/technoPolis/ml/bigData2020/hw0/Khomenko/src/main/resources/AB_NYC_2019.csv")

//median
data
  .groupBy("room_type")
  .agg(
    callUDF("percentile_approx", col("price"), lit(0.5)).as("median")
  )
  .show()

//mode
data
  .groupBy("room_type", "price")
  .count()
  .withColumn("row_number", row_number().over(Window.partitionBy("room_type").orderBy(desc("count"))))
  .select("room_type", "price")
  .where(col("row_number") === 1)
  .show()

//avg
data
  .groupBy("room_type")
  .agg(
    avg("price").as("avg")
  )
  .show()

//variance
data
  .select("room_type", "price")
  .groupBy("room_type")
  .agg(variance("price")).as("variance")
  .show()

//min price
data.orderBy("price").show(1)

//max price
data.orderBy(desc("price")).show(1)

//correlation price and minimum_nights
data.agg(
  corr("price", "minimum_nights")
)
  .show()

//correlation price and number_of_reviews
data.agg(
  corr("price", "number_of_reviews")
)
  .show()


//square 5х5 km with max avg price of apartament
//https://github.com/mumoshu/geohash-scala импортировать geohash либу для scala не удалось
val encodeGeoHash = (lat: Double, lng: Double, precision: Int) => {
  val base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
  var (minLat, maxLat) = (-90.0, 90.0)
  var (minLng, maxLng) = (-180.0, 180.0)
  val bits = List(16, 8, 4, 2, 1)

  (0 until precision).map { p => {
    base32 apply (0 until 5).map { i => {
      if (((5 * p) + i) % 2 == 0) {
        val mid = (minLng + maxLng) / 2.0
        if (lng > mid) {
          minLng = mid
          bits(i)
        } else {
          maxLng = mid
          0
        }
      } else {
        val mid = (minLat + maxLat) / 2.0
        if (lat > mid) {
          minLat = mid
          bits(i)
        } else {
          maxLat = mid
          0
        }
      }
    }
    }.reduceLeft((a, b) => a | b)
  }
  }.mkString("")
}

val geoHash_udf = udf(encodeGeoHash)

data
  .withColumn("geoHash", geoHash_udf(col("latitude"), col("longitude"), lit(5)))
  .groupBy("geoHash")
  .agg(
    avg("price").as("avg price")
  )
  .orderBy(desc("avg price"))
  .show(1)