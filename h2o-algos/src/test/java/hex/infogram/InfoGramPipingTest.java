package hex.infogram;

import org.junit.BeforeClass;
import org.junit.Test;
import water.DKV;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;

import static hex.infogram.InfoGramModel.InfoGramParameter;
import static hex.infogram.InfoGramModel.InfoGramParameter.Algorithm;

public class InfoGramPipingTest extends TestUtil {
  
  @BeforeClass public static void setup() { stall_till_cloudsize(1); }
  
  @Test
  public void testInfoGramInvoke() {
    try {
      Scope.enter();
      Frame trainF = parseTestFile("smalldata/glm_test/binomial_20_cols_10KRows.csv");
      convertCols(trainF);  // convert integer columns to enum columns
      Scope.track(trainF);
      DKV.put(trainF);
      
      InfoGramParameter params = new InfoGramParameter();
      params._response_column = "C21";
      params._train = trainF._key;
      params._algorithm = Algorithm.gbm;
      params._algorithm_params = "{\"sample_rate\" : [0.3], \"col_sample_rate\" : [0.3]}";
      params._ntop = 5;
      
      InfoGramModel infogramModel = new InfoGram(params).trainModel().get();
      Scope.track_generic(infogramModel);
      
    } finally {
      Scope.exit();
    }
  }
  
  public static void convertCols(Frame train) {
    final int numCols = train.numCols();
    final int enumCols = (numCols-1)/2;
    for (int index=0; index < enumCols; index++)
      train.replace(index, train.vec(index).toCategoricalVec()).remove();
    final int responseIndex = numCols-1;
    train.replace(responseIndex, train.vec(responseIndex).toCategoricalVec()).remove();
  }
}
