
clip_bnd <- function(vec, UB=1, LB=0) pmax(LB, pmin(vec, UB))

#' Run nade
#'
#' @description applies Nade
#'
#' @param texts vector of strings.
#'
#' @return A dataframe with intensities for eight basic emotions.
#' @export
#'
#' @examples
#' hello(c('I love pizza'))

nade <- function(txts) {
    
    # preprocessing
    txts <- tolower(txts)
    txts <- gsub("\r?\n|\r", " ", txts)
    
    # apply fasttext
    ft_model <- fastTextR::ft_load(
        system.file("data", "nade_250k_hp.ftz", package = "nadeR")
    )
    emoji_scores_raw <- fastTextR::ft_predict(ft_model, txts, k=151)
    
    emoji_jsn <- jsonlite::stream_in(
        file(system.file("data", "emoji_frequencies.jsonl", package = "nadeR")), 
        verbose = F
    )
    emoji_lookup <- dplyr::mutate(
        emoji_jsn, 
        label = paste0('__label__', hash)
    )
    
    sorted_preds <- dplyr::select(dplyr::arrange(
         dplyr::inner_join(emoji_scores_raw, emoji_lookup, by = 'label'),
         id.x, id.y
    ), id.x, id.y, prob)
    
    pred_matrix <- as.matrix(
        dplyr::select(tidyr::pivot_wider(
            sorted_preds, names_from = id.y, 
            values_from = prob
        ), -id.x)
    )
    
    m_anger <- lightgbm::lgb.load(system.file("data", "reg_anger", package = "nadeR"))
    m_fear <- lightgbm::lgb.load(system.file("data", "reg_fear", package = "nadeR"))
    m_trust <- lightgbm::lgb.load(system.file("data", "reg_trust", package = "nadeR"))
    m_anticipation <- lightgbm::lgb.load(system.file("data", "reg_anticipation", package = "nadeR"))
    m_disgust <- lightgbm::lgb.load(system.file("data", "reg_disgust", package = "nadeR"))
    m_sadness <- lightgbm::lgb.load(system.file("data", "reg_sadness", package = "nadeR"))
    m_joy <- lightgbm::lgb.load(system.file("data", "reg_joy", package = "nadeR"))
    m_surprise <- lightgbm::lgb.load(system.file("data", "reg_surprise", package = "nadeR"))

    data.frame(
        anger = clip_bnd(predict(m_anger, pred_matrix, type = "raw")),
        fear = clip_bnd(predict(m_fear, pred_matrix, type = "raw")),
        trust = clip_bnd(predict(m_trust, pred_matrix, type = "raw")),
        anticipation = clip_bnd(predict(m_anticipation, pred_matrix, type = "raw")),
        disgust = clip_bnd(predict(m_disgust, pred_matrix, type = "raw")),
        sadness = clip_bnd(predict(m_sadness, pred_matrix, type = "raw")),
        joy = clip_bnd(predict(m_joy, pred_matrix, type = "raw")),
        surprise = clip_bnd(predict(m_surprise, pred_matrix, type = "raw"))
    )
}