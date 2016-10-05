import org.apache.spark._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.Row
import org.apache.log4j.{Level, Logger}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.types.{StructType,StructField,StringType}

object ALSpredict {

  def main(args: Array[String]) {

    val sparkConf = new SparkConf().setAppName("ALSpredict")
    val sc = new SparkContext(sparkConf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val hiveContext = new HiveContext(sc)
    import hiveContext.implicits._
    import hiveContext.sql

    val source = args(0) // "s3a://jccsvdata/test3.data"
    val target = args(1) // "s3a://jcrecommand/json1"

    val data = sc.textFile(source)
    val data_ratings = data.map(_.split(',') match{ case Array(user,item,rate) => Rating(user.toInt,item.toInt,rate.toDouble) })

    val model = ALS.trainImplicit(data_ratings, 10, 20, 0.01, 10, 0.1 )
    //val model = MatrixFactorizationModel.load(sc,"/var/model/cf")

    val usersProducts = data_ratings.map{ case Rating(user, product, rate) => (user, product)}
    val predictions = model.predict(usersProducts).map{ case Rating(user, product, rate) => ((user, product), rate)}
    val ratesAndPreds = data_ratings.map{ case Rating(user, product, rate) => ((user, product), rate)}.join(predictions)
    val MSE = ratesAndPreds.map{ case ((user, product), (r1, r2)) => val err = (r1 - r2)
        err * err}.mean()
    println("MSE: " + MSE)

    val users = data_ratings.map{ case Rating(user, product, rate) => (user)}.distinct().collect()
    var list = ArrayBuffer[String]()
    users.foreach{ element =>
      var sugg = model.recommendProducts(element, 20)
      var res = sugg.filter({case Rating(user, prod, rate) => rate < 0.5 })
      var res_prod = res.map{case Rating(user, prod, rate) => (prod)}.mkString(",")
      list += element+ "-" +res_prod
    }

    val struct = new StructType( Array(
        StructField("userid", StringType, nullable = true),
        StructField("news", StringType, nullable = true))
    )

    val listRdd = sc.makeRDD(list.toArray).map(_.split("-")).map(p => Row(p(0), p(1)))
    val listDf  = sqlContext.createDataFrame(listRdd, struct)
    listDf.write.format("json").save(target)

    //model.save(sc, "s3a://jccsvdata/model")
    //val sameModel = MatrixFactorizationModel.load(sc,"/var/model/cf")
    sc.stop()
    println(args(0))
    println(args(1))
  }

}

