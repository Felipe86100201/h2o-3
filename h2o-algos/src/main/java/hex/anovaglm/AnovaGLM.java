package hex.anovaglm;

import hex.DataInfo;
import hex.ModelBuilder;
import hex.ModelBuilderHelper;
import hex.ModelCategory;
import hex.glm.GLM;
import hex.glm.GLMModel;
import jsr166y.ForkJoinTask;
import jsr166y.RecursiveAction;
import water.DKV;
import water.Key;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;

import java.util.ArrayList;
import java.util.List;

import static hex.anovaglm.AnovaGLMUtils.*;
import static hex.gam.MatrixFrameUtils.GamUtils.keepFrameKeys;
import static hex.glm.GLMModel.GLMParameters;
import static hex.glm.GLMModel.GLMParameters.Family.gaussian;
import static hex.glm.GLMModel.GLMParameters.Family.tweedie;

public class AnovaGLM extends ModelBuilder<AnovaGLMModel, AnovaGLMModel.AnovaGLMParameters, AnovaGLMModel.AnovaGLMOutput> {
  public static final int NUMBER_OF_MODELS = 3;
  DataInfo _dinfo;
  String[] _predictorNames; // store names of two predictors
  int[] _degreeOfFreedom;
  String[] _modelNames; // store model description
  Frame _weightOffsetFrame = null; // Frame storing weight and offset columns if present
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

    DataInfo dinfo = new DataInfo(_train.clone(), _valid, 1, true, DataInfo.TransformType.NONE,
            DataInfo.TransformType.NONE,
            _parms.missingValuesHandling() == GLMModel.GLMParameters.MissingValuesHandling.Skip,
            _parms.imputeMissing(), _parms.makeImputer(), false, hasWeightCol(), hasOffsetCol(),
            hasFoldCol(), null);
    
    if (!(dinfo._nums == 0)) 
      error("_predictors", " all predictors must be categorical.");
    
    if (!(dinfo._cats == 2))
      error("_predictors", " there must be two and only two categorical predictors for AnovaGLM.");
    
    if (hasWeightCol() || hasOffsetCol()) {
      int numWeightOffsetVecs = (hasWeightCol() ? 1 : 0) + (hasOffsetCol() ? 1 : 0);
      ExtractWeigthOffsetVecs ewVec = new ExtractWeigthOffsetVecs(_parms, dinfo).doAll(numWeightOffsetVecs, Vec.T_NUM,
              dinfo._adaptedFrame);
      _weightOffsetFrame = ewVec.outputFrame(Key.make(), ewVec._colNames, null);
    }
    
    _degreeOfFreedom = new int[NUMBER_OF_MODELS];
    _degreeOfFreedom[0] = dinfo._adaptedFrame.vec(0).domain().length-1;
    _degreeOfFreedom[1] = dinfo._adaptedFrame.vec(1).domain().length-1;
    _degreeOfFreedom[2] = _degreeOfFreedom[0]*_degreeOfFreedom[1];
    
    _predictorNames = new String[NUMBER_OF_MODELS];
    _predictorNames[0] = dinfo._adaptedFrame.name(0);
    _predictorNames[1] = dinfo._adaptedFrame.name(1);
    _predictorNames[2] = _predictorNames[0]+"_"+_predictorNames[1];

    _modelNames = new String[NUMBER_OF_MODELS];
    _modelNames[0] = "GLM Model with predictors "+_predictorNames[0]+", "+_predictorNames[2];
    _modelNames[1] = "GLM Model with predictors "+_predictorNames[1]+", "+_predictorNames[2];
    _modelNames[2] = "GLM Model with predictors "+_predictorNames[0]+", "+_predictorNames[1]+"_"+_predictorNames[2];
    
    if (error_count() > 0)
      throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(AnovaGLM.this);
  }

  private class AnovaGLMDriver extends Driver {
    Key<AnovaGLMModel>[] _glmModels; // store all GLM models built
    String[][] _transformedColNames;   // stored column names for transformed columns
    Key<Frame>[] _transformedCols;  // store transformed column frame keys
    Frame[] _trainingFrames;        // store generated frames
    GLMParameters[] _glmParams;      // store GLMParameters needed to generate all the data
    GLM[] _glmBuilder;               // store GLM Builders to be build in parallel
    GLM[] _glmResults;
    Key<Frame> _allTransformedCols;

    public final void buildModel() {
      AnovaGLMModel model = null;
      try {
        _trainingFrames = buildTrainingFrames(_transformedCols);  // build up training frames
        addWeightOffsetRebalanceFrames();
        _dinfo = new DataInfo(_trainingFrames[NUMBER_OF_MODELS-1].clone(), _valid, 1, true,
                DataInfo.TransformType.NONE, DataInfo.TransformType.NONE, 
                _parms.missingValuesHandling() == GLMModel.GLMParameters.MissingValuesHandling.Skip,
                _parms.imputeMissing(), _parms.makeImputer(), false, hasWeightCol(), hasOffsetCol(),
                hasFoldCol(), null);;
        _glmParams = buildGLMParameters(_trainingFrames, _parms);
        model = new AnovaGLMModel(dest(), _parms, new AnovaGLMModel.AnovaGLMOutput(AnovaGLM.this, _dinfo));
        model.write_lock(_job);
        _job.update(1, "calling GLM to build GLM models ...");
        _glmBuilder = buildGLMBuilders(_glmParams);
        _glmResults = ModelBuilderHelper.trainModelsParallel(_glmBuilder, NUMBER_OF_MODELS);  // build GLM models
        _job.update(0, "extracting metrics from GLM models and building AnovaGLM outputs");
      } finally {
        final List<Key<Vec>> keep = new ArrayList<>();
        if (model != null) {
          if (_parms._save_transformed_framekeys) {
            keepFrameKeys(keep, _allTransformedCols);
            model._output._transformed_columns_key = _allTransformedCols.toString();
          } else {
            removeKeys(_transformedCols);
          }
        }
      }
    }

    private void addWeightOffsetRebalanceFrames() {
      for (int index = 0; index < NUMBER_OF_MODELS; index++) {
        _trainingFrames[index] = new Frame(rebalance(_trainingFrames[index], false, _result +
                ".temporary.train"));
        if (_weightOffsetFrame != null) // add weight/offset columns if needed
          _trainingFrames[index].add(_weightOffsetFrame);

        DKV.put(_trainingFrames[index]);
      }
      if (_parms._save_transformed_framekeys)
        _allTransformedCols = _trainingFrames[NUMBER_OF_MODELS - 1]._key;
    }

    /***
     * This method will transform the training frame such that the constraints on the GLM parameters will be satisfied.  
     * Refer to Wendy Doc for more detail explanation.
     */
    void generateTransformedColumns() {
      _transformedColNames = new String[NUMBER_OF_MODELS][];
      _transformedCols = new Key[NUMBER_OF_MODELS];
      transformIndividualPredictor();  // transform individual predictors only
      // transform interaction columns
      final Frame vecsTransform = extractVec(new int[]{0,1}, _dinfo._adaptedFrame);
      _transformedColNames[2] = generateTransformedColNames(vecsTransform);
      final Frame transformedVecs = new Frame();
      transformedVecs.add(DKV.getGet(_transformedCols[0])); // add first transformed predictors
      transformedVecs.add(DKV.getGet(_transformedCols[1])); // add second transformed predictors
      TransformInteractionColumns tInteractC = new TransformInteractionColumns(_degreeOfFreedom, 
              _transformedColNames[2]).doAll(_degreeOfFreedom[2], Vec.T_NUM, transformedVecs);
      Frame convertedCol = tInteractC.outputFrame(Key.make(), _transformedColNames[2], null);
      _transformedCols[2] = convertedCol._key;
    }
    
    void transformIndividualPredictor() {
      int[] catNAFills = _dinfo.catNAFill();
      int numModelsMinus1 = NUMBER_OF_MODELS-1;
      RecursiveAction[] transformColumns = new RecursiveAction[numModelsMinus1];
      Frame trainFrame = _dinfo._adaptedFrame;  // first two columns are our predictors
      final Frame vec2Transform = extractVec(new int[]{0, 1}, trainFrame);
      for (int actionIndex = 0; actionIndex < numModelsMinus1; actionIndex++) {
        _transformedColNames[actionIndex] = generateTransformedColNames(vec2Transform);
        transformColumns[actionIndex] = new ColumnTransformation(vec2Transform, _transformedColNames[actionIndex],
                actionIndex, catNAFills);
      }
      ForkJoinTask.invokeAll(transformColumns);
    }
    
    public class ColumnTransformation extends RecursiveAction {
      final Frame _origFrame;
      final String[] _columnNames;
      final int _colIndex;
      final int[] _catNAFills;
      
      public ColumnTransformation(Frame frame, String[] newColNames, int colIndex, int[] catNAFills) {
        _origFrame = frame;
        _columnNames = newColNames;
        _colIndex = colIndex;
        _catNAFills = catNAFills;
      }
      
      @Override
      protected void compute() {
        GenerateTransformColumns gtc = new GenerateTransformColumns(_columnNames, _origFrame, _parms.imputeMissing(),
        _catNAFills, _colIndex).doAll(_columnNames.length, Vec.T_NUM, _origFrame);
        Frame convertedCol = gtc.outputFrame(Key.make(), _columnNames, null);
        _transformedCols[_colIndex] = convertedCol._key;
        DKV.put(convertedCol);
      }
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
