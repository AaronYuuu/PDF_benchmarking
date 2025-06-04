#!/usr/bin/Rscript

library(argparser)
library(RJSONIO)
library(yaml)
library(hgvsParseR)

#parse command line arguments
ap <- arg_parser(
  "interpolate template with mockup values",
  name = "interpolate.R"
)
ap <- add_argument(ap,
  "template_file",
  help = "template file to be interpolated (.tex)"
)
ap <- add_argument(ap,
  "json_file",
  help = "json data file with the values to interpolate into the template (.json)"
)
ap <- add_argument(ap,
  "--outprefix",
  help = "the output file"
)
args <- parse_args(ap)
if (is.na(args$outprefix)) {
  args$outprefix <- sub("\\.tex$", "", basename(args$template_file))
}



# Read the template
lines <- readLines(args$template_file)
text <- paste(lines, collapse = "\n")
blurb_data <- read_yaml("data/text_pieces.yml")
# Read the json data
mock_data <- fromJSON(args$json_file)

source("scripts/sharedFunctions.r")

# Determine which plugin to load based on template filename
template_basename <- basename(args$template_file)
template_name <- sub("\\.tex$", "", template_basename)

# Map template names to their corresponding plugin files
template_to_plugin <- list(
  "fakeHospital1" = "scripts/hospital1.r",
  "fakeHospital2" = "scripts/hospital2.r"
)


# Load the specific plugin for this template if it exists
if (template_name %in% names(template_to_plugin)) {
  plugin_file <- template_to_plugin[[template_name]]
  if (file.exists(plugin_file)) {
    source(plugin_file)
    cat("Loaded plugin for template:", template_name, "from", basename(plugin_file), "\n")
  } else {
    cat("Warning: Plugin file not found:", plugin_file, "\n")
  }
} else {
  cat("No specific plugin found for template:", template_name, "- using default functions only\n")
}

#helper function to extract data labels from template
extract <- function(text, rx, capture = TRUE) {
  matches <- gregexpr(rx, text, perl = TRUE)
  pos <- matches[[1]]
  if (length(pos) == 1 && pos[[1]] == -1) {
    #no matches, return empty data frame
    return(data.frame(label = character(), start = integer(), end = integer()))
  }
  len <- attr(matches[[1]], "match.length")
  if (capture) {
    labels <- cbind(
      attr(matches[[1]], "capture.start"),
      attr(matches[[1]], "capture.length")
    ) |> apply(1, \(x) substr(text, x[[1]], x[[1]] + x[[2]] - 1))
  } else {
    labels <- NA
  }
  data.frame(label = labels, start = pos, end = pos + len - 1)
}

# helper function to escape special characters for latex
tex_escape <- function(str) {
  #fixed substitution with pipe support
  g <- function(x, s, r) gsub(s, r, x, fixed = TRUE)
  #auto-prepend backslash
  e <- function(x, s) g(x, s, paste0("\\", s))
  #sequentially apply all rules
  str |>
    g("\\", "\\textasciibackslash") |>
    g("~", "\\textasciitilde") |>
    g("^", "\\textasciicircum") |>
    e("%") |>
    e("&") |>
    e("$") |>
    e("#") |>
    e("_") |>
    e("{") |>
    e("}")
}



#extract iterator sections

rx_begin_iter <- "\\\\begin\\{dataiter\\}\\{([^}]+)\\}"
rx_end_iter <- "\\\\end\\{dataiter\\}"
iter_starts <- extract(text, rx_begin_iter)
iter_ends <- extract(text, rx_end_iter, capture = FALSE)
#assert that each dataiter begin also has an end
stopifnot(nrow(iter_starts) == nrow(iter_ends))

#split text into sections and iterators
text_sections <- list()
last_end <- 0
for (i in seq_len(nrow(iter_starts))) {
  text_sections[[paste0("text_", i)]] <-
    substr(text, last_end + 1, iter_starts[i, "start"] - 1)
  text_sections[[paste0("iter_", i, ":", iter_starts[i, "label"])]] <-
    substr(text, iter_starts[i, "end"] + 1, iter_ends[i, "start"] - 1)
  last_end <- iter_ends[i, "end"]
}
text_sections[[paste0("text_", i + 1)]] <-
  substr(text, last_end + 1, nchar(text))

#extract field positions in each section
rx_field <- "\\\\data\\{([^}]+)\\}"
section_fields <- lapply(text_sections, \(txt) extract(txt, rx_field))

#iterate over datasets
# for (uuid in names(mock_data)) {
outputs <- lapply(names(mock_data), \(uuid) {
  dataset <- mock_data[[uuid]]
  #perform interpolations
  inter_sections <- lapply(seq_along(text_sections), \(i) {
    section_name <- names(text_sections)[[i]]
    txt <- text_sections[[i]]
    fields <- section_fields[[i]]
    #if this is a regular text section:
    if (startsWith(section_name, "text_")) {
      for (j in seq_len(nrow(fields))) {
        label <- fields[j, "label"]
        marker <- paste0("\\data{", label, "}")
        if (label == "blurb") {
          # Use the plugin's blurb function if available,
          if (exists("long_blurb") && is.function(long_blurb)) {
            blurb <- long_blurb(dataset$variants)
          } else {
            blurb <- paste("No explanation is available, please contact lab.",
                           "This is a placeholder for the blurb.")
          }
          txt <- sub(marker, blurb, txt, fixed = TRUE)
        } else if (label == "summary_blurb") {
          blurb <- summary_blurb(dataset$variants)
          txt <- sub(marker, blurb, txt, fixed = "TRUE")
        } else if (!(label %in% names(dataset))) {
          cat("Skipping unsupported label: ", label, "\n")
          txt <- sub(marker, paste0("\\it{Missing} ", label),
                     txt, fixed = TRUE)
        } else {
          value <- tex_escape(dataset[[label]])
          txt <- sub(marker, value, txt, fixed = "TRUE")
        }
      }
      txt
    } else {
      #otherwise, if this is an iterator section:
      iter_type <- sub("^[^:]+:", "", section_name)
      subdatasets <- dataset[[iter_type]]
      rows <- list()
      for (k in seq_along(subdatasets)) {
        sds <- subdatasets[[k]]
        row <- txt
        for (j in seq_len(nrow(fields))) {
          label <- fields[j, "label"]
          marker <- paste0("\\data{", label, "}")
          if (!(label %in% names(sds))) {
            cat("Skipping unsupported label: ", label, "\n")
            row <- sub(marker, "MISSING DATA!", row, fixed = TRUE)
          } else {
            value <- tex_escape(sds[[label]])
            row <- sub(marker, value, row, fixed = TRUE)
          }
        }
        rows <- c(rows, row)
      }
      paste(rows, collapse = "")
    }
  })
  paste0(inter_sections, collapse = "")
}) |> setNames(names(mock_data))

#write outputs to file
for (uuid in names(outputs)) {
  outfile <- paste0(args$outprefix, "_", uuid, ".tex")
  cat(outputs[[uuid]], file = outfile)
}

cat("Done!")