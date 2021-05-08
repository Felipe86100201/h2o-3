package hex.anovaglm;

import hex.Model;
import hex.glm.GLM;
import water.DKV;
import water.Key;
import water.fvec.Frame;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

import static hex.anovaglm.AnovaGLMModel.AnovaGLMParameters;
import static hex.gam.MatrixFrameUtils.GamUtils.setParamField;
import static hex.glm.GLMModel.GLMParameters;

public class AnovaGLMUtils {
  /***
   * Extract columns with indices specified in colInd into a H2OFrame.
   * @param colInd: integer array denoting column indices to be extracted
   * @param trainFrame: H2OFrame from which columns are to be extracted
   * @return: H2O Frame containing indices specified in colInd only
   */
  public static Frame extractVec(int[] colInd, Frame trainFrame) {
    final Frame extractedFrame = new Frame();
    for (int colIndex : colInd)
      extractedFrame.add(trainFrame.name(colIndex), trainFrame.vec(colIndex));
    return extractedFrame;
  }

  public static String[] generateTransformedColNames(Frame vec2Transform, int colIndex) {
    if (colIndex >= 0) // one column to transform
      return transformOneCol(vec2Transform, colIndex);
    else
      return transformTwoCols(vec2Transform);
  }
  
  public static String[] transformOneCol(Frame vec2Transform, int colInd) {
    String[] domains = vec2Transform.vec(colInd).domain();
    String colName = vec2Transform.name(colInd);
    int degOfFreedom = domains.length-1;
    String[] newColNames = new String[degOfFreedom];
    for (int domainInd = 0; domainInd < degOfFreedom; domainInd++) 
      newColNames[domainInd] = colName+"_"+domains[domainInd];
    return newColNames;
  }
  
  public static String[] transformTwoCols(Frame vec2Transform) {
    String[] domains1 = vec2Transform.vec(0).domain();
    String[] domains2 = vec2Transform.vec(1).domain();
    String colName1 = vec2Transform.name(0);
    String colName2 = vec2Transform.name(1);
    int degOfFreedomC1 = domains1.length-1;
    int degOfFreedomC2 = domains2.length-1;
    String[] newColNames = new String[degOfFreedomC1*degOfFreedomC2];
    int colIndex = 0;
    for (int col1 = 0; col1 < degOfFreedomC1; col1++)
      for (int col2 = 0; col2 < degOfFreedomC2; col2++) {
        newColNames[colIndex++] = colName1+"_"+domains1[col1]+"_"+colName2+"_"+domains2[col2];
      }
    return newColNames;
  }
  
  public static void removeKeys(Key<Frame>[] keys) {
    for (Key oneKey : keys)
      DKV.remove(oneKey);
  }

  /***
   * This method will take the frame that contains transformed columns of predictor A, predictor B, interaction
   * of predictor A and B and generate new training frames that contains the following columns:
   * - transformed columns of predictor A, interaction of predictor A and B, response
   * - transformed columns of predictor B, interaction of predictor A and B, response
   * - transformed columns of predictor A, predictor B, response
   * - transformed columns of predictor A, predictor B, interaction of predictor A and B, response
   * 
   * @param transformedCols: contains frame key of frame containing transformed columns of predictor A, predictor B,
   *                      interaction of predictor A and B
   * @param numberOfModels: number of models to build.  For 2 factors, this should be 4.
   * @return Array of predictors for the four models.
   */
  public static Frame[] buildTrainingFrames(Key<Frame> transformedCols, int numberOfModels, 
                                            String[][] transformedColNames) {
    Frame[] trainingFrames = new Frame[numberOfModels];
    int numFrames2Build = numberOfModels-1;
    Frame allCols = DKV.getGet(transformedCols);  // contains all the transformed columns except response, weight/offset
    trainingFrames[numFrames2Build] = allCols.clone();

    int[][] predNums = new int[numFrames2Build][];
    predNums[0] = new int[]{0,2}; // first predictor and interaction 
    predNums[1] = new int[]{1,2}; // second predictor and interaction
    predNums[2] = new int[]{0,1}; // first and second predictors

    for (int index = 0; index < numFrames2Build; index++) {
      trainingFrames[index] = buildSpecificFrame(predNums[index], allCols, transformedColNames);
    }
    return trainingFrames;
  }
  
  public static Frame buildSpecificFrame(int[] predNums, Frame allCols, String[][] transformedColNames) {
    final Frame predVecs = new Frame();
    int numVecs = predNums.length;
    for (int index = 0; index < numVecs; index++) {
      int predVecNum = predNums[index];
      predVecs.add(allCols.subframe(transformedColNames[predVecNum]));
    }
    return predVecs;
  }
  
  public static GLMParameters[] buildGLMParameters(Frame[] trainingFrames, AnovaGLMParameters parms) {
    int numberOfModels = trainingFrames.length;
    GLMParameters[] glmParams = new GLMParameters[numberOfModels];
    List<String> anovaGLMOnlyList = Arrays.asList("save_transformed_framekeys", "type");
    for (int index = 0; index < numberOfModels; index++) {
      glmParams[index] = new GLMParameters();
      Field[] field1 = AnovaGLMParameters.class.getDeclaredFields();
      setParamField(parms, glmParams[index], false, field1, anovaGLMOnlyList);
      Field[] field2 = Model.Parameters.class.getDeclaredFields();
      setParamField(parms, glmParams[index], true, field2, Arrays.asList());
      glmParams[index]._train = trainingFrames[index]._key;
    }
    return glmParams;
  }
  
  public static GLM[] buildGLMBuilders(GLMParameters[] glmParams) {
    int numModel = glmParams.length;
    GLM[] builder = new GLM[numModel];
    for (int index = 0; index < numModel; index++)
      builder[index] = new GLM(glmParams[index]);
    return builder;
  }
}
