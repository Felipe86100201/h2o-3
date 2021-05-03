package hex.anovaglm;

import hex.DataInfo;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.glm.GLMModel;
import jsr166y.ForkJoinTask;
import jsr166y.RecursiveAction;
import water.DKV;
import water.Key;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;

import static hex.anovaglm.AnovaGLMUtils.extractVec;
import static hex.anovaglm.AnovaGLMUtils.generateTransformedColNames;
import static hex.glm.GLMModel.GLMParameters.Family.gaussian;
import static hex.glm.GLMModel.GLMParameters.Family.tweedie;

public class AnovaGLM extends ModelBuilder<AnovaGLMModel, AnovaGLMModel.AnovaGLMParameters, AnovaGLMModel.AnovaGLMOutput> {
  public static final int NUMBER_OF_MODELS = 3;
  DataInfo _dinfo;
  String[] _predictorNames; // store names of two predictors
  int[] _degreeOfFreedom;
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
      _dinfo = new DataInfo(_train.clone(), _valid, 1, true, DataInfo.TransformType.NONE, 
              DataInfo.TransformType.NONE, 
              _parms.missingValuesHandling() == GLMModel.GLMParameters.MissingValuesHandling.Skip, 
              _parms.imputeMissing(), _parms.makeImputer(), false, hasWeightCol(), hasOffsetCol(), 
              hasFoldCol(), null);
      validateAnovaGLMParameters();
      _degreeOfFreedom = new int[NUMBER_OF_MODELS];
      _degreeOfFreedom[0] = _dinfo._adaptedFrame.vec(0).domain().length-1;
      _degreeOfFreedom[1] = _dinfo._adaptedFrame.vec(1).domain().length-1;
      _degreeOfFreedom[2] = _degreeOfFreedom[0]*_degreeOfFreedom[1];
    }
  }

  private void validateAnovaGLMParameters() {
    if (gaussian != _parms._family && tweedie != _parms._family)
      error("_family", " only gaussian and tweedie families are supported here");

    if (_parms._link == null)
      _parms._link = GLMModel.GLMParameters.Link.family_default;
    
    if (!(_dinfo._nums == 0)) 
      error("_predictors", " all predictors must be categorical.");
    
    if (!(_dinfo._cats == 2))
      error("_predictors", " there must be two and only two categorical predictors for AnovaGLM.");

    if (error_count() > 0)
      throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(AnovaGLM.this);
  }

  private class AnovaGLMDriver extends Driver {
    Key<AnovaGLMModel>[] _glmModels; // store all GLM models built
    String[][] _transformedColNames;   // stored column names for transformed columns
    Key<Frame>[] _transformedCols;  // store transformed column frame keys

    /***
     * This method will transform the training frame such that the constraints on the GLM parameters will be satisfied.  
     * Refer to Wendy Doc for more detail explanation.
     * @return H2OFrame with transformed frame
     */
    Frame[] adaptFrame() {
      Frame[] newTFrames = new Frame[NUMBER_OF_MODELS];
      _transformedColNames = new String[NUMBER_OF_MODELS][];
      _transformedCols = new Key[NUMBER_OF_MODELS];
      transformFrame();  // transform individual columns only
      
      return newTFrames;
    }
    
    void transformFrame() {
      int[] catNAFills = _dinfo.catNAFill();
      int numModelsMinus1 = NUMBER_OF_MODELS-1;
      RecursiveAction[] transformColumns = new RecursiveAction[numModelsMinus1];
      Frame trainFrame = _dinfo._adaptedFrame;  // first two columns are our predictors
      int[] colInd; // store column indices to extraction from trainFrame
      for (int actionIndex = 0; actionIndex < numModelsMinus1; actionIndex++) {
        colInd = new int[]{actionIndex};
        final Frame vec2Transform = extractVec(colInd, trainFrame);
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
      Frame[] newTrainFrames = adaptFrame();
    }
  }
}
