package info.rewiring.spark.mllib

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FunSpec

class LinearRegressionTest extends FunSpec{
  val sparkConf = new SparkConf()
    .setAppName("Linear regression")
    .setMaster("local")

  val sc = new SparkContext(sparkConf)
  val data = sc.textFile("file:///opt/spark/data/mllib/popvsrev/ex1data1.csv")

  val numOfIters = 50000
  //  val learnRate = 0.17339
  //  val learnRate = 0.17339121
  //  val learnRate = 0.1733912059
  //  val learnRate = 0.173391205906  // 19.54041891947169 (for 20) [-3.483043258547784,1.1511731089009738]
  //  val learnRate = 0.1733912059052 // 18.523431937340987 [-2.46748531945596,1.0495458628398473]
  //  val learnRate = 0.1733912059055 // 18.446650711355087 [-2.391405318002467,1.0419028014678777]
  //    val learnRate = 0.173391205906003 // 19.151030304332963 (for 20) [-3.0972108362943827,1.1124120570313674]
  //    val learnRate = 0.17339120590603 // 17.999775954854723 [-1.9486117615369702,0.9974193858195847]
  //    val learnRate = 0.1733912059060034 // 19.453797089404848 (for 20) [-3.3972125248916543,1.142550480714825]
//  val learnRate = 0.1733912059060035 // 19.323511349550316
//  val learnRate = 0.17339120590600341 // 19.453797089404848 (for 20) [-3.3972125248916543,1.142550480714825]
//  val learnRate = 0.17339120590600343 // 18.462801655762526 [-2.4074087597060245,1.0435105207734274]
  val learnRate = 0.1733912059060034176 //

  describe("Simple linear regression") {

    describe("Basic load, univariate train and predict tests") {
      it("Should load the provided resources correctly") {
        val expectedResults = Array("17.592,6.1101")
        assert(data.take(1).sameElements(expectedResults))
      }

      it("should create the correct LabeledPoint for the given data") {
        val linearRegression = LinearRegression(data, numOfIters, learnRate)

        val labeledPoint = linearRegression.getData()

        assert(labeledPoint.count() === 97)
        assert(labeledPoint.first().label === 17.592 )
        assert(labeledPoint.first().features === Vectors.dense(1, 6.1101))
      }

      it("should correctly train the model") {
        val linearRegression = LinearRegression(data, numOfIters, learnRate)
        val labeledPoint: RDD[LabeledPoint] = linearRegression.getData()

        val model: LinearRegressionModel = linearRegression.train(labeledPoint)


        println("Weights ------> " + model.weights)
        println("Intercept ------> " + model.intercept)
        linearRegression.evaluate(labeledPoint, model)
        assert(model.predict(Vectors.dense(1, 20.0)) === 19.453797089404848)
        assert(model.predict(Vectors.dense(1, 40.0)) === 42.304806703701345)
        assert(model.predict(Vectors.dense(1, 50.0)) === 53.730311510849596)

      }

    }

  }

  override def finalize(): Unit = {
    sc.stop()
  }


}
