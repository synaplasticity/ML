package info.rewiring.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FunSpec

class LinearRegressionTest extends FunSpec{
  val sparkConf = new SparkConf()
    .setAppName("Word count")
    .setMaster("local")

  val sc = new SparkContext(sparkConf)
  val data = sc.textFile("file:///opt/spark/data/mllib/popvsrev/ex1data1.csv")

  describe("Simple linear regression") {

    describe("Basic load, univariate train and predict tests") {
      it("Should load the provided resources correctly") {
        val expectedResults = Array("6.1101,17.592")
        assert(data.take(1).sameElements(expectedResults))
      }

    }

  }

}
