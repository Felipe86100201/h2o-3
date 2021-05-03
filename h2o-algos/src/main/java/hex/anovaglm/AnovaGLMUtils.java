package hex.anovaglm;

import water.fvec.Frame;

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

  public static String[] generateTransformedColNames(Frame vec2Transform) {
    if (vec2Transform.numCols() == 1) // one column to transform
      return transformOneCol(vec2Transform);
    else
      return transformTwoCols(vec2Transform);
  }
  
  public static String[] transformOneCol(Frame vec2Transform) {
    String[] domains = vec2Transform.vec(0).domain();
    String colName = vec2Transform.name(0);
    int degOfFreedom = domains.length-1;
    String[] newColNames = new String[degOfFreedom];
    for (int colIndex = 0; colIndex < degOfFreedom; colIndex++) 
      newColNames[colIndex] = colName+"_"+domains[colIndex];
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
}
