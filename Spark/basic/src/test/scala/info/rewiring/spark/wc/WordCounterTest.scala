package info.rewiring.spark.wc


import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FunSpec}

class WordCounterTest extends FunSpec with BeforeAndAfter {
  val sparkConf = new SparkConf()
    .setAppName("Word count")
    .setMaster("local")
  val sc = new SparkContext(sparkConf)
  val doc = sc.textFile("file:///opt/spark/data/wc/shakespeare.txt")


  describe("Word count test suite") {
    val wc: Words = new Words(doc)
    val flattened = wc.getFlattened(doc)
    val kvpairs = wc.getPairs(flattened)
    val countByWords = wc.getCountedReveresedOrder(kvpairs)
    val wordCount = wc.count

    describe("Basic internal step tests") {
      it("should flatten the RDD and return a list of words") {
        val expectedResults: Array[String] = Array("A", "MIDSUMMER", "NIGHT", "S", "DREAM")

        assert(flattened.take(5).sameElements(expectedResults))

      }

      it("should return a collection of pairs = (word, 1)") {
        val expectedResults: Array[(String, Int)] = Array(("a", 1), ("midsummer", 1))
        assert(kvpairs.take(2).sameElements(expectedResults))
      }


      it("should return a collection of pairs, wherein for a word a aggregate is provided") {
        val expectedResults: Array[(String, Int)] = Array(("zwaggered", 1), ("zur", 2))

        assert(countByWords.take(2).sameElements(expectedResults))
        assert(countByWords.count() == 22996)
      }

      it("should return the correct count of top 2 words") {
        val expectedResults: Array[(Int, String)] = Array((26856, "the"))
        assert(wordCount.take(1).sameElements(expectedResults))
      }
    }


  }

  override def finalize(): Unit = {
    sc.stop()
  }

}
