package hex.anovaglm;

import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;

public class GenerateTransformColumns extends MRTask<GenerateTransformColumns> {
  final public String[] _newColNames;
  final public int _newColNumber;
  final public boolean _transform1Col;
  public long _nobs;
  public int _degOfFreedom;
  final public boolean _imputeMissing;
  final public int[] _catNAFills;
  final public int _colIndex;
  
  public GenerateTransformColumns(String[] newColNames, Frame origFrame, boolean imputeMissing, int[] catNAFills, 
                                  int colIndex) {
    _newColNames = newColNames;
    _newColNumber = newColNames.length;
    _transform1Col = (origFrame.numCols()==1);
    _degOfFreedom = origFrame.vec(colIndex).domain().length-1;
    _imputeMissing = imputeMissing;
    _catNAFills = catNAFills;
    _colIndex = colIndex;
  }
  
  @Override
  public void map(Chunk[] chk, NewChunk[] newChk) {
    int numChkRows = chk[0].len();
    double[] changedRow = new double[_newColNumber];  // pre-allocate array for reuse
    for (int rowInd = 0; rowInd < numChkRows; rowInd++) {
        int val = readCatVal(chk, rowInd, _colIndex);
        if (val < 0)
          continue; // NAs are skipped here
        transformOneCol(changedRow, val, _degOfFreedom);
      for (int colInd = 0; colInd < _degOfFreedom; colInd++)
        newChk[colInd].addNum(changedRow[colInd]);
      _nobs++;
    }
  }
  
  @Override
  public void reduce(GenerateTransformColumns gt) {
    _nobs += gt._nobs;
  }
  
  public static void transformOneCol(double[] newRow, int val, int degOfFreedom) {
    for (int index = 0; index < degOfFreedom; index++) {
      if (val == index)
        newRow[index] = 1;
      else if (val == degOfFreedom)
        newRow[index] = -1;
      else 
        newRow[index] = 0;
    }
  }

  public int readCatVal(Chunk[] chk, int rowInd, int colInd) {
    double val = chk[colInd].atd(rowInd);
    if (Double.isNaN(val))
      if (_imputeMissing)
        return _catNAFills[colInd];
      else
        return -1;
    else
      return (int) val;
  }
}
