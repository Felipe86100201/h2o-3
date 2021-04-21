package hex.infogram;

import hex.Model;
import hex.ModelBuilder;
import hex.SplitFrame;
import hex.deeplearning.DeepLearningModel;
import hex.glm.GLMModel;
import hex.tree.drf.DRFModel;
import hex.tree.gbm.GBMModel;
import water.DKV;
import water.Key;
import water.fvec.Frame;
import water.util.TwoDimTable;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static hex.infogram.InfoGramModel.InfoGramParameter;
import static hex.infogram.InfoGramModel.InfoGramParameter.Algorithm;
public class InfoGramUtils {
  
  /**
   * This method will take the columns of _parms.train().  It will then remove the response, any columns in 
   * _parms._sensitive_attributes from the columns of _parms.train(), weights_column, offset_column.  Then, the 
   * columns that are left are the columns that are eligible to get their InfoGram.
   * 
   * @param _parms
   * @return
   */
  public static String[] extractPredictors(InfoGramParameter _parms) {
    Set<String> colNames = new HashSet<>(Arrays.asList(_parms.train().names()));
    Set<String> excludeCols = new HashSet<>(Arrays.asList(_parms._response_column));
    if (!(_parms._sensitive_attributes == null))
      excludeCols.addAll(new HashSet<>(Arrays.asList(_parms._sensitive_attributes)));  // remove sensitive attributes
    if (_parms._weights_column != null)
      excludeCols.addAll(new HashSet<>(Arrays.asList(_parms._weights_column)));  // remove sensitive attributes
    if (_parms._offset_column != null)
      excludeCols.addAll(new HashSet<>(Arrays.asList(_parms._offset_column)));  // remove sensitive attributes
    
    colNames.removeAll(excludeCols);
    String[] elligibleCols = new String[colNames.size()];
    colNames.toArray(elligibleCols);
    return elligibleCols;
  }
  
  public static String[] extractTopKPredictors(InfoGramParameter parms, Frame trainFrame, String[] eligiblePredictors) {
    if (parms._ntop > trainFrame.numCols()) return eligiblePredictors;
    ModelBuilder builder = ModelBuilder.make(parms._algorithm_parameters);
    Model builtModel = (Model) builder.trainModel().get();
    TwoDimTable varImp = extractVarImp(parms._algorithm, builtModel);
    String[] ntopPredictors = new String[parms._ntop];
    String[] rowHeaders = varImp.getRowHeaders();
    System.arraycopy(rowHeaders, 0, ntopPredictors, 0, parms._ntop);
    return ntopPredictors;
  }
  
  public static TwoDimTable extractVarImp(Algorithm algo, Model model) {
    switch (algo) {
      case gbm : return ((GBMModel) model)._output._variable_importances;
      case glm : return ((GLMModel) model)._output._variable_importances;
      case deeplearning : return ((DeepLearningModel) model)._output._variable_importances;
      case drf : return ((DRFModel) model)._output._variable_importances;
      default : return null;
    }
  }

  /**
   * This method will perform two functions:
   * - if user only wants a fraction of the training dataset to be used for infogram calculation, we will split the
   *   training frame and only use a fraction of it for infogram training purposes;
   * - next, a new training dataset will be generated containing only the predictors in predictors2Use array. 
   *
   * @param parms
   * @param predictors2Use
   * @param dataFraction
   * @param frame2Extract
   * @return
   */
  public static Frame extractTrainingFrame(InfoGramParameter parms, String[] predictors2Use, double dataFraction, 
                                           Frame frame2Extract) {
    Frame trainFrame = frame2Extract;
    if (dataFraction < 1) {  // only use a fraction training data for speedup
      SplitFrame sf = new SplitFrame(trainFrame, new double[]{parms._data_fraction, 1-parms._data_fraction}, 
              new Key[]{Key.make("train.hex"), Key.make("discard.hex")});
      sf.exec().get();
      Key[] ksplits = sf._destination_frames;
      trainFrame = DKV.get(ksplits[0]).get();
    }
    final Frame extractedFrame = new Frame(Key.make());
    for (String colName : predictors2Use) {
      extractedFrame.add(colName, trainFrame.vec(colName));
    }
    List<String> colNames = Arrays.asList(trainFrame.names());
    if (parms._weights_column != null && colNames.contains(parms._weights_column))
      extractedFrame.add(parms._weights_column, trainFrame.vec(parms._weights_column));
    if (parms._offset_column != null && colNames.contains(parms._offset_column))
      extractedFrame.add(parms._offset_column, trainFrame.vec(parms._offset_column));
    if (parms._response_column != null && colNames.contains(parms._response_column))
      extractedFrame.add(parms._response_column, trainFrame.vec(parms._response_column));

    DKV.put(extractedFrame);
    return extractedFrame;
  }
  
  public static void buildAlgorithmParameters(InfoGramParameter parms) {
    
  }
  
  public static void copyAlgoParams(ModelBuilder builder, InfoGramParameter parms) {
    String inputAlgoName = parms._algorithm.name();
    String algoName = ModelBuilder.algoName(inputAlgoName);
    String schemaDir = ModelBuilder.schemaDirectory(inputAlgoName);
    int algoVersion = 3;
    if (algoName.equals("SVD") || algoName.equals("Aggregator") || algoName.equals("StackedEnsemble")) algoVersion=99;
    String paramSchemaName = schemaDir+algoName+"V"+algoVersion+"$"+ModelBuilder.paramName(inputAlgoName)+"V"+algoVersion;
  }
}
