package hex.infogram;

import hex.ModelBuilder;
import hex.ModelCategory;
import water.Key;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;

import java.util.Arrays;
import java.util.List;

import static hex.infogram.CMI.calCoreInfoGram;
import static hex.infogram.CMI.calFairInfoGram;
import static hex.infogram.InfoGramModel.InfoGramParameter.Algorithm.AUTO;
import static hex.infogram.InfoGramModel.InfoGramParameter.Algorithm.gbm;
import static hex.infogram.InfoGramUtils.extractPredictors;
import static hex.infogram.InfoGramUtils.extractTopKPredictors;
import static hex.infogram.InfoGramUtils.extractTrainingFrame;

public class InfoGram extends ModelBuilder<InfoGramModel, InfoGramModel.InfoGramParameter,
        InfoGramModel.InfoGramOutput> {
  boolean _buildCore; // true to find core predictors, false to find admissible predictors
  String[] _topkPredictors;
  public InfoGram(boolean startup_once) { super(new InfoGramModel.InfoGramParameter(), startup_once);}
  
  public InfoGram(InfoGramModel.InfoGramParameter parms) {
    super(parms);
    init(false);
  }
  
  public InfoGram(InfoGramModel.InfoGramParameter parms, Key<InfoGramModel> key) {
    super(parms, key);
    init(false);
  }
  
  @Override
  protected Driver trainModelImpl() {
    return new IfoGramDriver();
  }

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[] { ModelCategory.Binomial, ModelCategory.Multinomial};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Experimental;
  }
  
  @Override public boolean havePojo() { return false; }
  @Override public boolean haveMojo() { return false; }
  
  @Override
  public void init(boolean expensive) {
    super.init(expensive);
    if (expensive)
      validateInfoGramParameters();
  }
  
  private void validateInfoGramParameters() {
    Frame dataset = _parms.train();
    // make sure sensitive_attributes are true predictor columns
    if (_parms._sensitive_attributes != null) {
      List<String> colNames = Arrays.asList(dataset.names());
      for (String senAttribute : _parms._sensitive_attributes)
        if (!colNames.contains(senAttribute))
          error("sensitive_attributes", "sensitive attribute: "+senAttribute+" is not a valid " +
                  "column in the training dataset.");
      _buildCore = false;
    } else {
      _buildCore = true;
    }
    // make sure conditional_info threshold is between 0 and 1
    if (_parms._conditional_info_threshold < 0 || _parms._conditional_info_threshold > 1)
      error("conditional_info_thresold", "conditional info threshold must be between 0 and 1.");
    
    // make sure varimp threshold is between 0 and 1
    if (_parms._varimp_threshold < 0 || _parms._varimp_threshold > 1)
      error("varimp_threshold", "varimp threshold must be between 0 and 1.");
    
    // check top k to be between 0 and training dataset column number
    if (_parms._ntop < 0 || _parms._ntop >= _parms.train().numCols())
      error("_topk", "topk must be between 0 and the number of predictor columns in your training dataset.");
    if (AUTO.equals(_parms._algorithm))
      _parms._algorithm = gbm;
    if (_parms._nfolds > 1)
      error("nfolds", "please specify nfolds as part of the algorithm specific parameter in " +
              "_algorithm_parms");
  }
  
  private class IfoGramDriver extends Driver {

    @Override
    public void computeImpl() {
      init(true);
      if (error_count() > 0)
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(InfoGram.this);
      _job.update(0, "Initializing model training");
      buildModel();
    }
    // todo:  add max_runtime_secs restrictions
    public final void buildModel() {
      InfoGramModel model = null;
      String[] eligiblePredictors = extractPredictors(_parms);  // exclude senstive attributes if applicable
      Frame trainingFrame = extractTrainingFrame(_parms, eligiblePredictors, _parms._data_fraction, _parms.train());
      _parms.fillImpl(); // copy over model specific parameters to _algorithm_parameters
      if (_parms._valid != null)
        _parms._algorithm_parameters._valid = extractTrainingFrame(_parms, eligiblePredictors, 1, _parms.valid()).getKey();
      _parms._algorithm_parameters._train = trainingFrame._key;
      _topkPredictors = extractTopKPredictors(_parms, trainingFrame, eligiblePredictors); // run model to extract ntop predictors
      if (_parms._sensitive_attributes == null) 
        calFairInfoGram();
      else
        calCoreInfoGram();
        

/*      String inputAlgoName = _parms._algorithm.toString();
      String algoName = ModelBuilder.algoName(inputAlgoName);
      String schemaDir = ModelBuilder.schemaDirectory(inputAlgoName);
      int algoVersion = 3;
      if (algoName.equals("SVD") || algoName.equals("Aggregator") || algoName.equals("StackedEnsemble")) algoVersion=99;
      String paramSchemaName = schemaDir+algoName+"V"+algoVersion+"$"+ModelBuilder.paramName(inputAlgoName)+"V"+algoVersion;*/
      
    }
  }
}
