package hex.anovaglm;

import water.MRTask;
import water.fvec.Chunk;
import water.fvec.NewChunk;

import java.util.stream.IntStream;

/***
 * This class will take two predictors and transform them according to rules specified in Wendy Docs
 */
public class GenerateTransformColumns extends MRTask<GenerateTransformColumns> {
  final public String[][] _newColNames;
  final public int[] _newColNumber;
  final public boolean _imputeMissing;
  final public int[] _catNAFills;
  final int _numNewCols;
  final int _numPreds;
  final int _totNewColNums;
  
  public GenerateTransformColumns(String[][] newColNames, boolean imputeMissing, int[] catNAFills) {
    _newColNames = newColNames;
    _newColNumber = new int[]{newColNames[0].length, newColNames[1].length, newColNames[2].length};
    _imputeMissing = imputeMissing;
    _catNAFills = catNAFills;
    _numNewCols = _newColNumber.length;
    _numPreds = _numNewCols-1;
    _totNewColNums = IntStream.of(_newColNumber).sum();
  }
  
  @Override
  public void map(Chunk[] chk, NewChunk[] newChk) {
    int numChkRows = chk[0].len();
    int[][] changedRow = allocateRow(_newColNumber);  // pre-allocate array for reuse
    int[] oneRow = new int[2];            // read in chunk row
    for (int rowInd = 0; rowInd < numChkRows; rowInd++) {
        if (!readCatVal(chk, rowInd, oneRow))
          continue; // imputeMissing=skip and encounter NAs

        transformOneRow(changedRow, oneRow, _numPreds, _newColNumber);
        int colIndex=0;
      for (int predInd = 0; predInd < _numNewCols; predInd++)
        for (int eleInd = 0; eleInd < _newColNumber[predInd]; eleInd++)
          newChk[colIndex++].addNum(changedRow[predInd][eleInd]);
    }
  }
  
  public static int[][] allocateRow(int[] newColNumber) {
    int numPreds = newColNumber.length;
    int[][] oneRow = new int[numPreds][];
    for (int index = 0; index < numPreds; index++)
      oneRow[index] = new int[newColNumber[index]];
    return oneRow;
  }
  
  public static void transformOneRow(int[][] newRow, int[] val, int numPreds, int[] newColNumber) {
    for (int rowInd = 0; rowInd < numPreds; rowInd++) { // work on each predictor and then 
      for (int valInd = 0; valInd < newColNumber[rowInd]; valInd++) {
        if (val[rowInd] == valInd)
          newRow[rowInd][valInd] = 1;
        else if (val[rowInd] == newColNumber[rowInd])
          newRow[rowInd][valInd] = -1;
        else
          newRow[rowInd][valInd] = 0;
      }
    }
    // create the interaction columns
    int countInteract = 0;
    for (int catInd1 = 0; catInd1 < newColNumber[0]; catInd1++) {
      for (int catInd2 = 0; catInd2 < newColNumber[1]; catInd2++)
        newRow[2][countInteract++] = newRow[0][catInd1]*newRow[1][catInd2];
    }
  }

  boolean readCatVal(Chunk[] chk, int rowInd, int[] rowData) {
    double val1 = chk[0].atd(rowInd);
    double val2 = chk[1].atd(rowInd);
    if (!_imputeMissing && (Double.isNaN(val1) || Double.isNaN(val2))) {  // skip row with any NAs with no imputeMissing
      return false;
    }
    // always impute missing
    if (Double.isNaN(val1))
      rowData[0] = _catNAFills[0];
    else
      rowData[0] = (int) val1;
    
    if (Double.isNaN(val2))
      rowData[1] = _catNAFills[1];
    else
      rowData[1] = (int) val2;

    return true;
  }
}
