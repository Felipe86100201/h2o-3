package hex.anovaglm;

import hex.DataInfo;
import water.MRTask;
import water.fvec.Chunk;
import water.fvec.NewChunk;

import static hex.anovaglm.AnovaGLMModel.AnovaGLMParameters;

public class ExtractWeigthOffsetVecs extends MRTask<ExtractWeigthOffsetVecs> {
  final AnovaGLMParameters _params;
  final DataInfo _dinfo;
  final int _weightChunkID;
  final int _offsetChunkID;
  final boolean _imputeMissing;
  final boolean _addWeight;
  final boolean _addOffset;
  String[] _colNames;
  
  public ExtractWeigthOffsetVecs(AnovaGLMParameters parms, DataInfo dinfo) {
    _params = parms;
    _dinfo = dinfo;
    _weightChunkID = dinfo._weights ? _dinfo.weightChunkId() : -1;
    _addWeight = _weightChunkID >= 0;
    _offsetChunkID = dinfo._offset ? _dinfo.offsetChunkId() : -1;
    _addOffset = _offsetChunkID >= 0;
    _imputeMissing = parms.imputeMissing();
    if (_addWeight && _addOffset) 
      _colNames = new String[]{parms._weights_column, parms._offset_column};
    else if (_addWeight && !_addOffset)
      _colNames = new String[]{parms._weights_column};
    else if (!_addWeight && _addOffset) 
      _colNames = new String[]{parms._offset_column};
  }

  @Override
  public void map(Chunk[] chk, NewChunk[] newChk) {
    int numRows = chk[0].len();
    for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
      if (_imputeMissing || !checkNAinRow(chk, rowIndex)) {
        int colInd = 0;
        if (_addWeight) 
          newChk[colInd++].addNum(chk[_weightChunkID].atd(rowIndex));
        if (_addOffset)
          newChk[colInd].addNum(chk[_offsetChunkID].atd(rowIndex));
      }
    }
  }

  public boolean checkNAinRow(Chunk[] chk, int rowInd) {
    double val = chk[0].atd(rowInd);
    double val2 = chk[1].atd(rowInd);
    return (Double.isNaN(val) || Double.isNaN(val2));
  }
}
