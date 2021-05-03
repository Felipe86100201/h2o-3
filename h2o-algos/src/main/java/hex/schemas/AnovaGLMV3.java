package hex.schemas;

import hex.anovaglm.AnovaGLM;
import hex.anovaglm.AnovaGLMModel;
import hex.glm.GLMModel;
import water.api.API;
import water.api.schemas3.KeyV3;
import water.api.schemas3.ModelParametersSchemaV3;

public class AnovaGLMV3 extends ModelBuilderSchema<AnovaGLM, AnovaGLMV3, AnovaGLMV3.AnovaGLMParametersV3> {

  public static final class AnovaGLMParametersV3 extends ModelParametersSchemaV3<AnovaGLMModel.AnovaGLMParameters,
          AnovaGLMParametersV3> {
    public static final String[] fields = new String[] {
            "model_id",
            "training_frame",
            "seed",
            "response_column",
            "ignored_columns",
            "ignore_const_cols",
            "score_each_iteration",
            "offset_column",
            "weights_column",
            "family",
            "tweedie_variance_power",
            "tweedie_link_power",
            "theta", // equals to 1/r and should be > 0 and <=1, used by negative binomial
            "solver",
            "missing_values_handling",
            "plug_values",
            "compute_p_values",
            "non_negative",
            "max_iterations",
            "link",
            "prior",
            "stopping_rounds",
            "stopping_metric",
            "early_stopping",
            "stopping_tolerance",
            // dead unused args forced here by backwards compatibility, remove in V4
            "balance_classes",
            "class_sampling_factors",
            "max_after_balance_size",
            "max_runtime_secs",
            "type" // GLM SS Type, only support 3 right now
    };

    @API(help = "Seed for pseudo random number generator (if applicable)", gridable = true)
    public long seed;

    // Input fields
    @API(help = "Family. Use binomial for classification with logistic regression, others are for regression problems.",
            values = {"AUTO", "gaussian", "tweedie"}, level = API.Level.critical)
    // took tweedie out since it's not reliable
    public GLMModel.GLMParameters.Family family;

    @API(help = "Tweedie variance power", level = API.Level.critical, gridable = true)
    public double tweedie_variance_power;

    @API(help = "Tweedie link power", level = API.Level.critical, gridable = true)
    public double tweedie_link_power;

    @API(help = "Theta", level = API.Level.critical, gridable = true)
    public double theta; // used by negtaive binomial distribution family

    @API(help = "AUTO will set the solver based on given data and the other parameters. IRLSM is fast on on problems" +
            " with small number of predictors and for lambda-search with L1 penalty, L_BFGS scales better for datasets" +
            " with many columns.", values = {"AUTO", "IRLSM", "L_BFGS","COORDINATE_DESCENT_NAIVE",
            "COORDINATE_DESCENT", "GRADIENT_DESCENT_LH", "GRADIENT_DESCENT_SQERR"}, level = API.Level.critical)
    public GLMModel.GLMParameters.Solver solver;

    @API(help = "Handling of missing values. Either MeanImputation, Skip or PlugValues.", values = { "MeanImputation",
            "Skip", "PlugValues" }, level = API.Level.expert, direction=API.Direction.INOUT, gridable = true)
    public GLMModel.GLMParameters.MissingValuesHandling missing_values_handling;

    @API(help = "Plug Values (a single row frame containing values that will be used to impute missing values of the" +
            " training/validation frame, use with conjunction missing_values_handling = PlugValues)",
            direction = API.Direction.INPUT)
    public KeyV3.FrameKeyV3 plug_values;

    @API(help="Request p-values computation, p-values work only with IRLSM solver and no regularization",
            level = API.Level.secondary, direction = API.Direction.INPUT)
    public boolean compute_p_values; // _remove_collinear_columns

    @API(help = "Maximum number of iterations", level = API.Level.secondary)
    public int max_iterations;

    @API(help = "Link function.", level = API.Level.secondary, values = {"family_default", "identity", "logit", "log",
            "inverse", "tweedie", "ologit"}) //"oprobit", "ologlog": will be supported.
    public GLMModel.GLMParameters.Link link;

    @API(help = "Prior probability for y==1. To be used only for logistic regression iff the data has been sampled and" +
            " the mean of response does not reflect reality.", level = API.Level.expert)
    public double prior;

    // dead unused args, formely inherited from supervised model schema
    /**
     * For imbalanced data, balance training data class counts via
     * over/under-sampling. This can result in improved predictive accuracy.
     */
    @API(help = "Balance training data class counts via over/under-sampling (for imbalanced data).",
            level = API.Level.secondary, direction = API.Direction.INOUT)
    public boolean balance_classes;

    /**
     * Desired over/under-sampling ratios per class (lexicographic order).
     * Only when balance_classes is enabled.
     * If not specified, they will be automatically computed to obtain class balance during training.
     */
    @API(help = "Desired over/under-sampling ratios per class (in lexicographic order). If not specified, sampling" +
            " factors will be automatically computed to obtain class balance during training. Requires " +
            "balance_classes.", level = API.Level.expert, direction = API.Direction.INOUT)
    public float[] class_sampling_factors;

    /**
     * When classes are balanced, limit the resulting dataset size to the
     * specified multiple of the original dataset size.
     */
    @API(help = "Maximum relative size of the training data after balancing class counts (can be less than 1.0). " +
            "Requires balance_classes.", /* dmin=1e-3, */ level = API.Level.expert, direction = API.Direction.INOUT)
    public float max_after_balance_size;

    @API(help = "Refer to the SS type 1, 2, 3, or 4.  We are currently only supporting 3", level = API.Level.critical)
    public int type;  // GLM SS Type, only support 3

    @API(help="Stop early when there is no more relative improvement on train or validation (if provided)")
    public boolean early_stopping;
  }
}
