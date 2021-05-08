package hex.anovaglm;

import hex.glm.GLMModel;
import org.junit.Test;
import org.junit.runner.RunWith;
import water.DKV;
import water.Scope;
import water.fvec.Frame;
import water.runner.CloudSize;
import water.runner.H2ORunner;

import static hex.glm.GLMModel.GLMParameters.Family.gaussian;
import static water.TestUtil.parseTestFile;

@RunWith(H2ORunner.class)
@CloudSize(1)
public class AnovaGLMBasicTest {

  /**
   * Test to check that training frame has been transformed correctly
   */
  @Test
  public void testFrameTransform() {
    try {
      Scope.enter();
      Frame correctFrame = parseTestFile("smalldata/anovaglm/MooreTransformed.csv");
      Frame train = parseTestFile("smalldata/anovaglm/Moore.csv");
      Scope.track(correctFrame);
      Scope.track(train);

      AnovaGLMModel.AnovaGLMParameters params = new AnovaGLMModel.AnovaGLMParameters();
      params._family = gaussian;
      params._response_column = "conformity";
      params._train = train._key;
      params._solver = GLMModel.GLMParameters.Solver.IRLSM;
      params._ignored_columns = new String[]{"fscore"};
      params._save_transformed_framekeys = true;
      AnovaGLMModel anovaG = new AnovaGLM(params).trainModel().get();
      Scope.track_generic(anovaG);
      Frame transformedFrame = DKV.getGet(anovaG._output._transformedColumnKey);
      Scope.track(transformedFrame);
    } finally {
      Scope.exit();
    }
  }

  @Test
  public void testWeightOffset() {
    try {
      Scope.enter();
      Frame train = parseTestFile("smalldata/extdata/prostate.csv");
      train.replace(1, train.vec(1).toCategoricalVec()).remove();
      train.replace(3, train.vec(3).toCategoricalVec()).remove();
      DKV.put(train);
      Scope.track(train);

      AnovaGLMModel.AnovaGLMParameters params = new AnovaGLMModel.AnovaGLMParameters();
      params._family = gaussian;
      params._response_column = "VOL";
      params._train = train._key;
      params._solver = GLMModel.GLMParameters.Solver.IRLSM;
      params._weights_column = "AGE";
      params._offset_column = "GLEASON";
      params._ignored_columns = new String[]{"ID", "DPROS", "DCAPS", "PSA"};
      params._save_transformed_framekeys = true;

      AnovaGLMModel anovaG = new AnovaGLM(params).trainModel().get();
      Scope.track_generic(anovaG);
    } finally {
      Scope.exit();
    }
  }
}
