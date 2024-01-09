# Readme

This package allows to infer basic emotions from social media messages. While human raters are often too resource-intensive, lexical approaches face challenges regarding incomplete vocabulary and the handling of informal language. Even advanced machine learning-based approaches require substantial resources (expert knowledge, programming skills, annotated data sets, extensive computational capabilities) and tend to gauge the mere presence, not the intensity, of emotion. This package solves this issue by predicting a vast array of emojis based on the surrounding text, then reduces these predicted emojis to an established set of eight basic emotions.

## Installation

```R
devtools::install_github('inkrement/nader')
```

## Usage

```R
library(nadeR)

n <- nade()
n$load()
n$predict('I love pizza')
```

## Known Issues

 - [ ] fasttext package in R does not correclty apply our ova-loss [EmilHvitfeldt/fastTextR#8](https://github.com/EmilHvitfeldt/fastTextR/issues/8)