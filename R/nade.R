
# ---------------------------------------------------------------------
# clip_bnd
# ========
#' @title Clip Bounds
#' @description Clip bounds of a vector
#' 
#' @param vec vector to clip
#' @param UB upper bound
#' @param LB lower bound
#' 
clip_bnd <- function(vec, UB=1, LB=0) {
    pmax(LB, pmin(vec, UB))
}

# ---------------------------------------------------------------------
# long_to_wide_baseR
# ==================
#' @title Long to wide
#'  
#' @param sorted_preds sorted predictions
#' 
#' @return A data frame of wide predictions
#' 
#' @importFrom reshape2 dcast
#' 
long_to_wide_baseR <- function(sorted_preds) {
  wide_data <- reshape2::dcast(
    data = sorted_preds,
    formula = id ~ int_label,
    value.var = "prob"
  )
  wide_data <- wide_data[,-1]  # Remove the first column (id.x)
  colnames(wide_data) <- gsub("prob.", "", colnames(wide_data))  # Remove the prefix "prob."
  return(wide_data)
}


# ---------------------------------------------------------------------
# nadeR_load
# ==========
#' @title Load nadeR model
#' @description Load nadeR model
#' 
#' @return A list of models
#' @export
#' 
nadeR_load <- function() {
    model <- nade()

    model$update_model(
        structure(
            list(
                ft_model = fastTextR::ft_load(
                    system.file("data", "nade_250k_hp.ftz", package = "nadeR")
                ),
                emoji_lookup = dplyr::mutate(
                    jsonlite::stream_in(
                        file(system.file("data", "emoji_frequencies.jsonl", package = "nadeR")), 
                        verbose = F
                    ), 
                    label = paste0('__label__', hash)
                ),
                m_anger = lightgbm::lgb.load(system.file("data", "reg_anger", package = "nadeR")),
                m_fear = lightgbm::lgb.load(system.file("data", "reg_fear", package = "nadeR")),
                m_trust = lightgbm::lgb.load(system.file("data", "reg_trust", package = "nadeR")),
                m_anticipation = lightgbm::lgb.load(system.file("data", "reg_anticipation", package = "nadeR")),
                m_disgust = lightgbm::lgb.load(system.file("data", "reg_disgust", package = "nadeR")),
                m_sadness = lightgbm::lgb.load(system.file("data", "reg_sadness", package = "nadeR")),
                m_joy = lightgbm::lgb.load(system.file("data", "reg_joy", package = "nadeR")),
                m_surprise = lightgbm::lgb.load(system.file("data", "reg_surprise", package = "nadeR"))
            ),
            class = "nade"
        )
    )

    model
}

# hotfix
#
#' @importFrom fastText fasttext_interface
#' 
ft_hotfix_predict <- function(txts) {
    tmp_x_data = tempfile(fileext = '.txt')
    tmp_out_data = tempfile(fileext = '.txt')
    writeLines(text = txts, con = tmp_x_data, sep = '\n')

    list_params <- list(
        command = 'predict-prob',
        model = system.file("data", "nade_250k_hp.ftz", package = "nadeR"),
        test_data = tmp_x_data,
        k = 151,
        th = 0.0
    )

    res = fasttext_interface(
        list_params,
        path_output = tmp_out_data
    )

    tmp_predictions <- readLines(tmp_out_data)

    tmp_predictions

    predictions_scores <- do.call(rbind, lapply(tmp_predictions, function(predictions) {
          data.frame( 
            label = unlist(strsplit(predictions, split=' '))[seq(1,302,2)],
            prob = unlist(strsplit(predictions, split=' '))[seq(2,302,2)]) 
        }
     )
     )
    predictions_scores$id <- rep(1:length(tmp_predictions), each = 151)
    return(predictions_scores)
}


# ---------------------------------------------------------------------
# nadeR_predict
# =============
#' @title Predict emotions
#' @description Predict emotions
#' 
#' @param model nadeR model
#' @param txts text to predict
#' 
#' @return A data frame of emotions
#' 
#' @examples
#' model <- nadeR_load()
#' nadeR_predict(model, "I am happy")
#' 
##' @importFrom fastTextR ft_predict
#' @export 
nadeR_predict <- function(model, txts) {
    # preprocessing
    txts <- tolower(txts)
    txts <- gsub("\r?\n|\r", " ", txts)
    
    # apply fasttext
    # ... some dirty hotfix, cause it's R
    #emoji_scores_raw <- fastTextR::ft_predict(model$ft_model, txts, k=151)
    emoji_scores_raw <- ft_hotfix_predict(txts)

    emoji_scores_raw$int_label <- as.integer(substring(emoji_scores_raw$label, 10))
    sorted_preds <- emoji_scores_raw[order(emoji_scores_raw$id, emoji_scores_raw$int_label),]
    
    pred_matrix <- as.matrix(
        long_to_wide_baseR(sorted_preds)
    )
    
    data.frame(
        anger = clip_bnd(predict(model$m_anger, pred_matrix, type = "raw")),
        fear = clip_bnd(predict(model$m_fear, pred_matrix, type = "raw")),
        trust = clip_bnd(predict(model$m_trust, pred_matrix, type = "raw")),
        anticipation = clip_bnd(predict(model$m_anticipation, pred_matrix, type = "raw")),
        disgust = clip_bnd(predict(model$m_disgust, pred_matrix, type = "raw")),
        sadness = clip_bnd(predict(model$m_sadness, pred_matrix, type = "raw")),
        joy = clip_bnd(predict(model$m_joy, pred_matrix, type = "raw")),
        surprise = clip_bnd(predict(model$m_surprise, pred_matrix, type = "raw"))
    )
}


# ---------------------------------------------------------------------
# nade
# ====
#' @title Create a New \code{nade} Object
#' @description Create a new \code{nade} model. The avalable methods are:
#' \itemize{
#'  \item \code{load} - load the model
#' }
#' 
#' @examples 
#' model <- nade()
#' model$load()
#' 
#' @return A new \code{nade} object
#' 
#' @export
nade <- function() {
    model <- new.env(parent = emptyenv())

    model$ft_model <- NULL
    model$emoji_lookup <- NULL 
    model$m_anger <- NULL
    model$m_fear <- NULL
    model$m_trust <- NULL
    model$m_anticipation <- NULL
    model$m_disgust <- NULL
    model$m_sadness <- NULL
    model$m_joy <- NULL
    model$m_surprise <- NULL

    model$update_model <- function(new_model) {
        self <- parent.env(environment())$model

        self$ft_model <- new_model$ft_model
        self$emoji_lookup <- new_model$emoji_lookup

        self$m_anger <- new_model$m_anger
        self$m_fear <- new_model$m_fear
        self$m_trust <- new_model$m_trust
        self$m_anticipation <- new_model$m_anticipation
        self$m_disgust <- new_model$m_disgust
        self$m_sadness <- new_model$m_sadness
        self$m_joy <- new_model$m_joy
        self$m_surprise <- new_model$m_surprise

        class(self) <- class(new_model)

        self
    }
    
    model$load <- function() {
        self <- parent.env(environment())$model
        self$update_model(nadeR_load())

        return(invisible(NULL))
    }

    model$predict <- function(txts) {
        self <- parent.env(environment())$model

        if (is.null(self$ft_model)) {
            stop("Model not loaded. Please run `load()` first.")
        }
        
        nadeR_predict(self, txts)
    }

    class(model) <- "nade"

    model
}


# ---------------------------------------------------------------------
# predict.nade
# ============
#' @title Predict emotions
#' @description Predict emotions
#' 
#' @param object nadeR model
#' @param txts text to predict
#' 
#' @return A data frame of emotions
#' 
#' @examples
#' model <- nadeR_load()
#' predict(model, "I am happy")
#' 
#' @export 
predict.nade <- function(object, txts) {
    object$predict(txts)
}