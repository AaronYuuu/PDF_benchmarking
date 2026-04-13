if (!require(ggplot2, quietly = TRUE)) {
    install.packages("ggplot2")
}
library(ggplot2)
options(stringsAsFactors = FALSE)

if (!require(remotes, quietly = TRUE)) {
    install.packages("remotes")
}
if (!require(yogitools, quietly = TRUE)) {
    remotes::install_github("jweile/yogitools", force = TRUE)
}
library(yogitools)

#dev.off()
library(dplyr)
library(tidyr)
library(lme4)
library(emmeans)

# Read the CSV file with full path
df <- read.csv("/Users/ayu/PDF_benchmarking/graphs/Hospitalfinal.csv")

# Create Input column based on LLM column
df$Input <- ifelse(grepl("image", df$LLM, ignore.case = TRUE), "Image", "Text")

# Fill NA values in Prompt column with "Normal"
df$Prompt[is.na(df$Prompt)] <- "Normal"
df$Prompt[df$Prompt == "None"] <- "zero-shot"

# Drop columns with 'unnamed' in the name (case insensitive)
unnamed_cols <- grep("unnamed", names(df), ignore.case = TRUE)
if (length(unnamed_cols) > 0) {
    df <- df[, -unnamed_cols]
}

# Count values in Input column
table(df$Input)

# -------------------------------
# Supplementary figure:
# Parsed-only accuracy (non-iTT)
# -------------------------------
df_parsed <- df %>%
    filter(Parsed == TRUE) %>%
    mutate(
        PromptClean = case_when(
            Prompt == "Normal" ~ "zero-shot",
            Prompt == "LTNER/GPT-NER" ~ "one-shot",
            Prompt == "None" ~ "zero-shot",
            TRUE ~ tolower(Prompt)
        ),
        InputClean = tolower(Input),
        LLMClean = gsub("\\*ImageInput\\*|\\s\\(One-shot\\)|\\s\\(Zero-shot\\)", "", LLM)
    )

acc_parsed_stats <- df_parsed %>%
    group_by(LLMClean, PromptClean, InputClean) %>%
    summarise(
        n_runs = n(),
        mean_accuracy_parsed = mean(Accuracy, na.rm = TRUE),
        sd_accuracy_parsed = sd(Accuracy, na.rm = TRUE),
        .groups = "drop"
    )

write.csv(
    acc_parsed_stats,
    "/Users/ayu/PDF_benchmarking/graphs/accuracy_parsed_only_summary.csv",
    row.names = FALSE
)

acc_plot <- ggplot(
    acc_parsed_stats,
    aes(x = LLMClean, y = mean_accuracy_parsed, fill = interaction(PromptClean, InputClean))
) +
    geom_col(position = position_dodge(width = 0.9)) +
    geom_errorbar(
        aes(ymin = pmax(0, mean_accuracy_parsed - sd_accuracy_parsed),
            ymax = pmin(100, mean_accuracy_parsed + sd_accuracy_parsed)),
        width = 0.2,
        position = position_dodge(width = 0.9)
    ) +
    labs(
        title = "Parsed-only Accuracy by Model and Condition",
        x = "Model",
        y = "Accuracy (%)",
        fill = "Prompt;Input"
    ) +
    coord_cartesian(ylim = c(0, 100)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(
    filename = "/Users/ayu/PDF_benchmarking/graphs/Supplementary_ParsedOnly_Accuracy.png",
    plot = acc_plot,
    width = 12,
    height = 6,
    dpi = 300
)

# Group by LLM, Prompt, and Input, then calculate mean and std of F1score
df_stats <- df %>%
    group_by(LLM, Prompt, Input) %>%
    summarise(
        n_runs = n(),
        parse_rate = mean(Parsed, na.rm = TRUE) * 100,
        mean = mean(F1score, na.rm = TRUE),
        std = sd(F1score, na.rm = TRUE),
        .groups = 'drop'
    )

# Optional parsed-only summary (legacy metric, not primary)
df_stats_parsed_only <- df %>%
    filter(Parsed == TRUE) %>%
    group_by(LLM, Prompt, Input) %>%
    summarise(
        mean_parsed_only = mean(F1score_ParsedOnly, na.rm = TRUE),
        std_parsed_only = sd(F1score_ParsedOnly, na.rm = TRUE),
        .groups = 'drop'
    )

# Save a compact summary table for manuscript reporting
write.csv(df_stats, "/Users/ayu/PDF_benchmarking/graphs/F1_parse_summary.csv", row.names = FALSE)

# Clean up LLM names
df_stats$LLM <- gsub("\\*ImageInput\\*", "", df_stats$LLM)
df_stats$LLM <- gsub(" \\(One-shot\\)", "", df_stats$LLM)
df_stats$LLM <- gsub(" \\(Zero-shot\\)", "", df_stats$LLM)

# Define name mapping function
name_mapping <- function(x) {
  case_when(
    grepl("gpt-4.1-mini", x) ~ "GPT-4.1",
        grepl("llama3.170b", x) ~ "Llama3.1",
        grepl("NuExtract:4B", x) ~ "NuExtract",
        grepl("mistral\\(24b\\)", x) ~ "Mistral",
        grepl("gemma327b", x) ~ "Gemma",
        x == "GLiNER" ~ "GliNER",
        grepl("biomed-base-v1.0", x) ~ "BioGliNER",
        TRUE ~ x
    )
}

# Apply name mapping
df_stats$LLM <- name_mapping(df_stats$LLM)

# View the results
print(df_stats)

# Convert df_stats to f1_data format for plotting
f1_data <- df_stats %>%
    rename(model = LLM, f1 = mean) %>%
    mutate(
        prompt = case_when(
            Prompt == "Normal" ~ "zero-shot",
            Prompt == "LTNER/GPT-NER" ~ "one-shot",
            Prompt == "None" ~ "zero-shot",
            TRUE ~ tolower(Prompt)
        ),
        input = tolower(Input)
    ) %>%
    select(model, f1, prompt, input)

combos <- expand.grid(
    list(prompt = c("zero-shot", "one-shot"), input = c("text", "image")), 
    stringsAsFactors = FALSE
)
combos$label <- apply(combos, 1, paste, collapse = ";")

# Get all unique models
mnames <- unique(f1_data$model)

# Create a complete grid to ensure all combinations exist
complete_grid <- expand.grid(
    model = mnames,
    prompt = c("zero-shot", "one-shot"),
    input = c("text", "image"),
    stringsAsFactors = FALSE
)

# Merge with actual data to fill missing combinations with NA
f1_complete <- merge(complete_grid, f1_data, all.x = TRUE)

# Create F1 matrix with consistent dimensions
f1_mat <- matrix(NA, nrow = length(mnames), ncol = nrow(combos))
rownames(f1_mat) <- mnames
colnames(f1_mat) <- combos$label

for (i in seq_along(mnames)) {
    for (j in seq_len(nrow(combos))) {
        combo <- combos[j, ]
        val <- f1_complete$f1[f1_complete$model == mnames[i] & 
                              f1_complete$prompt == combo$prompt & 
                              f1_complete$input == combo$input]
        if (length(val) > 0 && !is.na(val)) {
            f1_mat[i, j] <- val
        }
    }
}

# Create standard error matrix with consistent dimensions  
stderr_mat <- matrix(NA, nrow = length(mnames), ncol = nrow(combos))
rownames(stderr_mat) <- mnames
colnames(stderr_mat) <- combos$label

for (i in seq_along(mnames)) {
    for (j in seq_len(nrow(combos))) {
        combo <- combos[j, ]
        
        # More flexible prompt matching for different model types
        matching_rows <- df_stats[
            df_stats$LLM == mnames[i] & 
            tolower(df_stats$Input) == combo$input,
        ]
        
        # Filter by prompt type
        if (combo$prompt == "zero-shot") {
            matching_rows <- matching_rows[
                matching_rows$Prompt %in% c("Normal", "None", "zero-shot", "", NA) | 
                is.na(matching_rows$Prompt) |
                matching_rows$Prompt == "",
            ]
        } else if (combo$prompt == "one-shot") {
            matching_rows <- matching_rows[
                matching_rows$Prompt == "LTNER/GPT-NER",
            ]
        }
        
        if (nrow(matching_rows) > 0 && !is.na(matching_rows$std[1])) {
            stderr_mat[i, j] <- abs(matching_rows$std[1])  # Force positive error bars
        }
    }
}

# Add debugging output to see what's happening
cat("\nDebugging standard error calculation:\n")
cat("Models in mnames:", paste(mnames, collapse = ", "), "\n")

# Check specific models
for (model_name in c("GliNER", "BioGliNER", "NuExtract")) {
    if (model_name %in% mnames) {
        model_data <- df_stats[df_stats$LLM == model_name, ]
        cat("\n", model_name, " data:\n")
        print(model_data)
        
        # Check stderr_mat for this model
        model_row <- which(mnames == model_name)
        cat("Standard errors for", model_name, ":", stderr_mat[model_row, ], "\n")
    }
}

# Use the stderr_mat as stderr
stderr <- stderr_mat

# Calculate real p-values using statistical tests
# Get raw F1 scores for each condition to perform statistical comparisons
raw_data_for_stats <- df %>%
    mutate(
        prompt_clean = case_when(
            Prompt == "Normal" ~ "zero-shot",
            Prompt == "LTNER/GPT-NER" ~ "one-shot", 
            Prompt == "None" ~ "zero-shot",
            TRUE ~ tolower(Prompt)
        ),
        input_clean = tolower(Input),
        llm_clean = name_mapping(gsub("\\*ImageInput\\*|\\s\\(One-shot\\)|\\s\\(Zero-shot\\)", "", LLM))
    )

# -------------------------------
# Primary inference:
# mixed-effects + emmeans contrasts
# -------------------------------
raw_data_for_stats <- raw_data_for_stats %>%
    mutate(
        # Robust boolean parsing for Parsed column
        Parsed = case_when(
            Parsed %in% c(TRUE, "TRUE", "True", 1, "1") ~ 1L,
            TRUE ~ 0L
        ),
        DocID = as.factor(DocID),
        llm_clean = as.factor(llm_clean),
        prompt_clean = as.factor(prompt_clean),
        input_clean = as.factor(input_clean),
        Distressed = as.factor(Distressed)
    )

# Mixed model for iTT F1 score (continuous primary quality endpoint)
f1_lmm <- lmer(
    F1score ~ llm_clean * prompt_clean * input_clean + Distressed + (1 | DocID),
    data = raw_data_for_stats
)

f1_emm <- emmeans(f1_lmm, ~ llm_clean | prompt_clean + input_clean)
f1_contrasts <- as.data.frame(pairs(f1_emm, adjust = "none"))
f1_contrasts$endpoint <- "F1score_iTT"
f1_contrasts$estimate_type <- "mean_difference"
f1_contrasts$p_value <- f1_contrasts$p.value
f1_contrasts$p_bh <- p.adjust(f1_contrasts$p_value, method = "BH")

# Mixed logistic model for parse success (reliability endpoint)
parse_glmm <- glmer(
    Parsed ~ llm_clean * prompt_clean * input_clean + Distressed + (1 | DocID),
    data = raw_data_for_stats,
    family = binomial(link = "logit")
)

parse_emm <- emmeans(parse_glmm, ~ llm_clean | prompt_clean + input_clean, type = "link")
parse_contrasts <- as.data.frame(pairs(parse_emm, adjust = "none"))
parse_contrasts$endpoint <- "ParseSuccess"
parse_contrasts$estimate_type <- "log_odds_difference"
parse_contrasts$odds_ratio <- exp(parse_contrasts$estimate)
parse_contrasts$p_value <- parse_contrasts$p.value
parse_contrasts$p_bh <- p.adjust(parse_contrasts$p_value, method = "BH")

# Harmonize and export primary stats table
f1_primary <- f1_contrasts %>%
    transmute(
        endpoint,
        prompt_clean,
        input_clean,
        contrast,
        estimate,
        SE,
        df,
        statistic = t.ratio,
        odds_ratio = NA_real_,
        estimate_type,
        p_value,
        p_bh
    )

parse_primary <- parse_contrasts %>%
    transmute(
        endpoint,
        prompt_clean,
        input_clean,
        contrast,
        estimate,
        SE,
        df = NA_real_,
        statistic = z.ratio,
        odds_ratio,
        estimate_type,
        p_value,
        p_bh
    )

stats_primary_table <- bind_rows(f1_primary, parse_primary)
write.csv(
    stats_primary_table,
    "/Users/ayu/PDF_benchmarking/graphs/stats_primary_table.csv",
    row.names = FALSE
)

# -------------------------------
# Sensitivity analysis:
# pairwise MWU (within prompt/input)
# -------------------------------
split_conditions <- split(
    raw_data_for_stats,
    list(raw_data_for_stats$prompt_clean, raw_data_for_stats$input_clean),
    drop = TRUE
)

mwu_rows <- list()

for (cond_name in names(split_conditions)) {
    cond_df <- split_conditions[[cond_name]]
    models_here <- sort(unique(as.character(cond_df$llm_clean)))
    if (length(models_here) < 2) next

    cond_parts <- strsplit(cond_name, "\\.")[[1]]
    prompt_val <- cond_parts[1]
    input_val <- cond_parts[2]

    combs <- combn(models_here, 2, simplify = FALSE)
    for (pair in combs) {
        m1 <- pair[1]
        m2 <- pair[2]
        x <- cond_df$F1score[cond_df$llm_clean == m1]
        y <- cond_df$F1score[cond_df$llm_clean == m2]
        x <- x[!is.na(x)]
        y <- y[!is.na(y)]
        if (length(x) < 2 || length(y) < 2) next

        wt <- wilcox.test(x, y, exact = FALSE)
        mwu_rows[[length(mwu_rows) + 1]] <- data.frame(
            endpoint = "F1score_iTT",
            prompt_clean = prompt_val,
            input_clean = input_val,
            contrast = paste0(m1, " - ", m2),
            n_model_1 = length(x),
            n_model_2 = length(y),
            median_model_1 = median(x),
            median_model_2 = median(y),
            median_diff = median(x) - median(y),
            p_value = wt$p.value
        )
    }
}

if (length(mwu_rows) > 0) {
    mwu_sensitivity <- bind_rows(mwu_rows)
    mwu_sensitivity$p_bh <- p.adjust(mwu_sensitivity$p_value, method = "BH")
    write.csv(
        mwu_sensitivity,
        "/Users/ayu/PDF_benchmarking/graphs/stats_mwu_sensitivity_table.csv",
        row.names = FALSE
    )
}

# Define colors
mycolors <- c("firebrick2", "firebrick4", "steelblue2", "steelblue4")

# Use tryCatch to ensure dev.off() is called even if errors occur
tryCatch({
    png("F1_scores_plot.png", width = 10, height = 5, units = "in", res = 600)
    # Set margins and axis label orientations
    op <- par(mar = c(2, 4, 4, 2), cex = 1.2)

    # Draw plot
    xs <- barplot(
        t(f1_mat), beside = TRUE, space = c(0, 1.5),
        ylim = c(0, 100),
        col = mycolors, border = NA, 
        ylab = expression(F[1] ~ "score (%)"), 
        main = "F1 Score by Model and Condition",
        xaxt = "n"
    )

    # Add custom rotated x-axis labels
    x_coords <- colMeans(xs) + 0.5
    # Adjust y position to be below the plot area
    y_pos <- par("usr")[3] - 2 
    text(x = x_coords, y = y_pos, labels = rownames(f1_mat), srt = 0, adj = 1, xpd = TRUE)


    # Draw the error bars using real standard errors
    yogitools::errorBars(xs, t(f1_mat), t(stderr), l = .05)

    # Determine x-coordinates of missing values
    xna <- xs[apply(t(f1_mat), 1:2, is.na)]

    # Add hatching at missing value positions
    if (length(xna) > 0) {
        rect(xna - 0.5, 0, xna + 0.5, 100, col = "gray", density = 20, border = NA)
    }

    # No significance annotations here; inferential results are exported
    # from the mixed-effects primary table and MWU sensitivity table.

    # Add grid lines
    grid(NA, NULL)

    # Add legend
    legend("topright", colnames(f1_mat), fill = mycolors, bg = "white")

    # Restore original parameters
    par(op)

}, finally = {
    # Ensure the PDF device is closed
    dev.off()
})

# Print matrices for verification
cat("\nF1 Score Matrix:\n")
print(f1_mat)
cat("\nStandard Error Matrix:\n")
print(stderr_mat)