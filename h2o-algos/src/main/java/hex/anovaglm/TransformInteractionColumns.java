package hex.anovaglm;

import water.MRTask;
import water.fvec.Chunk;
import water.fvec.NewChunk;

public class TransformInteractionColumns extends MRTask<TransformInteractionColumns> {
  final int[] _degreesOfFreedom;
  final String[] _transformedColNames;
  final int _numTransformedColumns;
  
  public TransformInteractionColumns(int[] degOfFreedom, String[] newColNames) {
    _degreesOfFreedom = degOfFreedom;
    _transformedColNames = newColNames;
    _numTransformedColumns = newColNames.length;
  }

  @Override
  public void map(Chunk[] chk, NewChunk[] newChk) {
    int[] rowValues = new int[_numTransformedColumns];
    int[] pred1Row = new int[_degreesOfFreedom[0]];
    int[] pred2Row = new int[_degreesOfFreedom[1]];
    int numRows = chk[0].len();
    for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
      readPredictorRow(pred1Row, pred2Row, rowIndex, chk); // read in predictors
      generateInteraction(pred1Row, pred2Row, rowValues);
      for (int colIndex = 0; colIndex < _degreesOfFreedom[2]; colIndex++)
        newChk[colIndex].addNum(rowValues[colIndex]);
    }
  }
  
  void readPredictorRow(int[] pred1, int[] pred2, int rowIndex, Chunk[] chk) {
    for (int index = 0; index < _degreesOfFreedom[0]; index++)
      pred1[index] = (int) chk[index].at8(rowIndex);
    for (int index = 0; index < _degreesOfFreedom[1]; index++)
      pred2[index] = (int) chk[index+_degreesOfFreedom[0]].atd(rowIndex);
  }
  
  void generateInteraction(int[] pred1Row, int[] pred2Row, int[] resultRow) {
    int resultInd = 0;
    for (int index1 = 0; index1 < _degreesOfFreedom[0]; index1++) 
      for (int index2 = 0; index2 < _degreesOfFreedom[1]; index2++)
        resultRow[resultInd++] = pred1Row[index1]*pred2Row[index2];
  }
  
}
