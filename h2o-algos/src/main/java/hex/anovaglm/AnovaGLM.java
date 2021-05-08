package hex.anovaglm;

import hex.DataInfo;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.glm.GLM;
import hex.glm.GLMModel;
import water.DKV;
import water.Key;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;

import java.util.ArrayList;
import java.util.List;

import static hex.anovaglm.AnovaGLMUtils.extractVec;
import static hex.anovaglm.AnovaGLMUtils.generateTransformedColNames;
import static hex.gam.MatrixFrameUtils.GamUtils.keepFrameKeys;
import static hex.glm.GLMModel.GLMParameters;
import static hex.glm.GLMModel.GLMParameters.Family.gaussian;
import static hex.glm.GLMModel.GLMParameters.Family.tweedie;
import static water.util.ArrayUtils.flat;

public class AnovaGLM extends ModelBuilder<AnovaGLMModel, AnovaGLMModel.AnovaGLMParameters, AnovaGLMModel.AnovaGLMOutput> {
  public static final int NUMBER_OF_MODELS = 4;// (A, A*B), (B, A*B), (A, B), (A, B, A*B)
  public static final int NUMBER_OF_PREDICTORS = 3; // A, B, interaction of A and B
  DataInfo _dinfo;
  String[] _predictorNames; // store names of two predictors
  int[] _degreeOfFreedom;
  String[] _modelNames; // store model description
  Frame _weightOffsetFrame = null; // Frame storing weight and offset columns if present
  public int[] _catNAFills;
  public AnovaGLM(boolean startup_once) { super (new AnovaGLMModel.AnovaGLMParameters(), startup_once); }

  public AnovaGLM(AnovaGLMModel.AnovaGLMParameters parms) {
    super(parms);
    init(false);
  }

  public AnovaGLM(AnovaGLMModel.AnovaGLMParameters parms, Key<AnovaGLMModel> key) {
    super(parms, key);
    init(false);
  }

  @Override
  protected int nModelsInParallel(int folds) {  // disallow nfold cross-validation
    return nModelsInParallel(1, 2);
  }

  @Override
  protected AnovaGLMDriver trainModelImpl() { return new AnovaGLMDriver(); }

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{ModelCategory.Regression};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  public boolean haveMojo() { return false; }

  @Override
  public boolean havePojo() { return false; }

  public BuilderVisibility buildVisibility() { return BuilderVisibility.Experimental; }

  public void init(boolean expensive) {
    super.init(expensive);
    if (expensive) {
      initValidateAnovaGLMParameters();
    }
  }

  /***
   * Init and validate AnovaGLMParameters.
   */
  private void initValidateAnovaGLMParameters() {
    if (gaussian != _parms._family && tweedie != _parms._family)
      error("_family", " only gaussian and tweedie families are supported for now.");

    if (_parms._link == null)
      _parms._link = GLMModel.GLMParameters.Link.family_default;

    _dinfo = new DataInfo(_train.clone(), _valid, 1, true, DataInfo.TransformType.NONE,
            DataInfo.TransformType.NONE,
            _parms.missingValuesHandling() == GLMModel.GLMParameters.MissingValuesHandling.Skip,
            _parms.imputeMissing(), _parms.makeImputer(), false, hasWeightCol(), hasOffsetCol(),
            hasFoldCol(), null);
    
    if (!(_dinfo._nums == 0)) 
      error("_predictors", " all predictors must be categorical.");
    
    if (!(_dinfo._cats == 2))
      error("_predictors", " there must be two and only two categorical predictors for AnovaGLM.");
    
    if (hasWeightCol() || hasOffsetCol()) {
      int numWeightOffsetVecs = (hasWeightCol() ? 1 : 0) + (hasOffsetCol() ? 1 : 0);
      ExtractWeigthOffsetVecs ewVec = new ExtractWeigthOffsetVecs(_parms, _dinfo).doAll(numWeightOffsetVecs, Vec.T_NUM,
              _dinfo._adaptedFrame);
      _weightOffsetFrame = ewVec.outputFrame(Key.make(), ewVec._colNames, null);
    }
    
    _catNAFills = _dinfo.catNAFill().clone();
    _degreeOfFreedom = new int[NUMBER_OF_MODELS];
    _degreeOfFreedom[0] = _dinfo._adaptedFrame.vec(0).domain().length-1;
    _degreeOfFreedom[1] = _dinfo._adaptedFrame.vec(1).domain().length-1;
    _degreeOfFreedom[2] = _degreeOfFreedom[0]*_degreeOfFreedom[1];
    _degreeOfFreedom[3] = _degreeOfFreedom[0]+_degreeOfFreedom[1];
    
    _predictorNames = new String[NUMBER_OF_PREDICTORS];
    _predictorNames[0] = _dinfo._adaptedFrame.name(0);
    _predictorNames[1] = _dinfo._adaptedFrame.name(1);
    _predictorNames[2] = _predictorNames[0]+"_"+_predictorNames[1];

    _modelNames = new String[NUMBER_OF_MODELS];
    _modelNames[0] = "GLM Model with predictors "+_predictorNames[0]+", "+_predictorNames[2];
    _modelNames[1] = "GLM Model with predictors "+_predictorNames[1]+", "+_predictorNames[2];
    _modelNames[2] = "GLM Model with predictors "+_predictorNames[0]+", "+_predictorNames[1];
    _modelNames[3] = "GLM Model with predictors "+_predictorNames[0]+", "+_predictorNames[1]+"_"+_predictorNames[2];
    
    if (error_count() > 0)
      throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(AnovaGLM.this);
  }

  private class AnovaGLMDriver extends Driver {
    Key<AnovaGLMModel>[] _glmModels; // store all GLM models built
    String[][] _transformedColNames;   // stored column names for transformed columns
    String[] _allTransformedColNames;   // flatten new column names
    Key<Frame> _transformedColsKey;  // store transformed column frame key
    Frame[] _trainingFrames;        // store generated frames
    GLMParameters[] _glmParams;      // store GLMParameters needed to generate all the data
    GLM[] _glmBuilder;               // store GLM Builders to be build in parallel
    GLM[] _glmResults;
    Frame _completeTransformedFrame;  // store transformed frame

    public final void buildModel() {
      AnovaGLMModel model = null;
      try {
        model = new AnovaGLMModel(dest(), _parms, new AnovaGLMModel.AnovaGLMOutput(AnovaGLM.this));
  /*              _dinfo = new DataInfo(_parms.train().clone(), _valid, 1, true,
                DataInfo.TransformType.NONE, DataInfo.TransformType.NONE,
                _parms.missingValuesHandling() == GLMModel.GLMParameters.MissingValuesHandling.Skip,
                _parms.imputeMissing(), _parms.makeImputer(), false, hasWeightCol(), hasOffsetCol(),
                hasFoldCol(), null);
        model = new AnovaGLMModel(dest(), _parms, new AnovaGLMModel.AnovaGLMOutput(AnovaGLM.this, _dinfo)); */
        model.write_lock(_job);
      //  model.delete_and_lock();
/*        _trainingFrames = buildTrainingFrames(_transformedColsKey, NUMBER_OF_MODELS, _transformedColNames);  // build up training frames
        addRespWeightOffsetRebalanceFrames();
        _glmParams = buildGLMParameters(_trainingFrames, _parms);
        _job.update(1, "calling GLM to build GLM models ...");
        _glmBuilder = buildGLMBuilders(_glmParams);
        _glmResults = ModelBuilderHelper.trainModelsParallel(_glmBuilder, NUMBER_OF_MODELS);  // build GLM models*/
        


        _job.update(0, "Completed GLM model building.  Extracting metrics from GLM models and building" +
                " AnovaGLM outputs");

        model.update(_job);
      } finally {
        final List<Key<Vec>> keep = new ArrayList<>();
        if (model != null) {
          if (_parms._save_transformed_framekeys) {
            keepFrameKeys(keep, _transformedColsKey);
            model._output._transformed_columns_key = _transformedColsKey.toString();
            model._output._transformedColumnKey = _transformedColsKey;
          } else {
            DKV.remove(_transformedColsKey);
          }
          model.unlock(_job);
        }
      }
    }

    private void addRespWeightOffsetRebalanceFrames() {
      for (int index = 0; index < NUMBER_OF_MODELS; index++) {
        Frame currentFrame = _trainingFrames[index];
        currentFrame.add(_parms._response_column, _parms.train().vec(_parms._response_column)); // add response
        if (_weightOffsetFrame != null) // add weight/offset columns if needed
          currentFrame.add(_weightOffsetFrame);
        _trainingFrames[index] = new Frame(rebalance(currentFrame, false, _result +
                ".temporary.train"));
        DKV.put(_trainingFrames[index]);
      }
    }

    /***
     * This method will transform the training frame such that the constraints on the GLM parameters will be satisfied.  
     * Refer to Wendy Doc for more detail explanation.
     */
    void generateTransformedColumns() {
      _transformedColNames = new String[NUMBER_OF_PREDICTORS][];
      int numOnlyPredictors = NUMBER_OF_PREDICTORS-1; // converting only predictors, no interaction
      Frame trainFrame = _dinfo._adaptedFrame;  // first two columns are our predictors
      final Frame vec2Transform = extractVec(new int[]{0, 1}, trainFrame);
      for (int actionIndex = 0; actionIndex < numOnlyPredictors; actionIndex++) {
        _transformedColNames[actionIndex] = generateTransformedColNames(vec2Transform, actionIndex);
      }
      _transformedColNames[numOnlyPredictors] = generateTransformedColNames(vec2Transform, -1);
      _allTransformedColNames = flat(_transformedColNames);
      GenerateTransformColumns gtc = new GenerateTransformColumns(_transformedColNames, _parms.imputeMissing(), 
              _catNAFills);
      gtc.doAll(gtc._totNewColNums, Vec.T_NUM,  vec2Transform);
      _completeTransformedFrame = gtc.outputFrame(Key.make(), _allTransformedColNames, null);
      _transformedColsKey = _completeTransformedFrame._key;
      DKV.put(_completeTransformedFrame);
    }
    
    @Override
    public void computeImpl() {
      init(true);
      generateTransformedColumns();
      _job.update(0, "Finished transforming training frame");
      buildModel();
    }
  }
}
