package info.rewiring.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FunSpec

class LinearRegressionTest extends FunSpec{
  val sparkConf = new SparkConf()
    .setAppName("Word count")
    .setMaster("local")

  val sc = new SparkContext(sparkConf)
  val data = sc.textFile("file:///opt/spark/data/mllib/popvsrev/ex1data1.csv")

  val numOfIters = 10
  val learnRate = 0.01

  describe("Simple linear regression") {

    describe("Basic load, univariate train and predict tests") {
      it("Should load the provided resources correctly") {
        val expectedResults = Array("6.1101,17.592")
        assert(data.take(1).sameElements(expectedResults))
      }

      it("should create the correct LabeledPoint for the given data") {
        val linearRegression = LinearRegression(data, numOfIters, learnRate)

        val labeledPoint = linearRegression.getData()

        assert(labeledPoint.count() === 97)
        assert(labeledPoint.first().label === 6.1101)
        assert(labeledPoint.first().features.toString === "[17.592]")
      }

    }

  }

  override def finalize(): Unit = {
    sc.stop()
  }


}
