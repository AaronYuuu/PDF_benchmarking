#!/usr/bin/env Rscript

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
  "json_file",
  help = paste(
    "json data file with the values to",
    "interpolate into the template (.json)"
  )
)
ap <- add_argument(ap,
  "--template_file",
  help = "override template file (.tex) to use instead of 
          automatically chosen ones"
)
ap <- add_argument(ap,
  "--template_dir",
  help = "the directory in the script will look for templates",
)
ap <- add_argument(ap,
  "--all_templates",
  help = "process all templates in the template directory",
  flag = TRUE
)
ap <- add_argument(ap,
  "--outprefix",
  help = "the output file",
  default = "report"
)
args <- parse_args(ap)
# if (is.na(args$outprefix)) {
#   args$outprefix <- sub("\\.tex$", "", basename(args$template_file))
# }

# Find the path to this script
script_path <- function() {
  (
    grep("--file", commandArgs(), fixed = TRUE, value = TRUE) |>
      strsplit("=", fixed = TRUE)
  )[[1]][[2]]
}


# Load the blurb generation text pieces
data_dir <- normalizePath(paste0(dirname(script_path()), "/../data"))
blurb_data <- read_yaml(paste0(data_dir, "/text_pieces.yml"))
if (is.na(args$template_dir)) {
  args$template_dir <- normalizePath(
    paste0(dirname(script_path()), "/../templates")
  )
}
if (!dir.exists(args$template_dir)) {
  stop("Template directory ", args$template_dir, " does not exist!")
}
# Load the template registry
templates <- unlist(read_yaml(paste0(data_dir, "/template_registry.yml")))
#validate templates
tmpl_files <- names(templates) |>
  lapply(\(tmpl) {
    paste0(args$template_dir, "/", tmpl, c(".tex", "_plugin.R"))
  }) |>
  unlist()
files_present <- file.exists(tmpl_files) |> setNames(tmpl_files)
if (!all(files_present)) {
  stop(
    "Missing template or plugin files: ",
    paste(names(which(!files_present)), collapse = ", ")
  )
}

# Read the json data
if (!file.exists(args$json_file)) {
  stop("JSON data file does not exist: ", args$json_file)
}
mock_data <- fromJSON(args$json_file)

#if a static template was providedd, check that the template file 
#exists and is a valid tex file
if (!is.na(args$template_file)) {
  if (!file.exists(args$template_file)) {
    stop("Template file does not exist: ", args$template_file)
  }
  if (!grepl("\\.tex$", args$template_file)) {
    stop("Template file must be a .tex file: ", args$template_file)
  }
}

####################
# HELPER FUNCTIONS #
####################

match_template <- function(lab_name) {
  template <- names(which(templates == lab_name))
  # FIXME: Add parameter for template directory
  paste0(args$template_dir, "/", template, ".tex")
}

#capitalize a word ("hello" -> "Hello")
cap <- function(txt) {
  substr(txt, 1, 1) <- toupper(substr(txt, 1, 1))
  txt
}
#convert a number to text (2 -> "two")
num2text <- function(num, one = FALSE) {
  if (num == 1 && !one) {
    return("a") # says "a" instead of "one"
  }
  blurb_data$numbers[[num]]
}
#convert an amino acid's code to its full name
aaname <- function(aa) {
  blurb_data$residues[[tolower(aa)]]
}

#helper function to extract data labels from template
# parameters:
# - text: the template text
# - rx: the regular expression to match
# - capture: whether to capture the label names (default TRUE)
# returns: a data frame with label names, start and end positions
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
    e("$") |>
    e("#") |>
    e("_") |>
    e("{") |>
    e("}")
}

# helper function to escape special characters for latex tables
# (does not escape & since it's used as column separator)
tex_escape_table <- function(str) {
  #fixed substitution with pipe support
  g <- function(x, s, r) gsub(s, r, x, fixed = TRUE)
  #auto-prepend backslash
  e <- function(x, s) g(x, s, paste0("\\", s))
  #sequentially apply all rules but skip &
  str |>
    g("\\", "\\textasciibackslash") |>
    g("~", "\\textasciitilde") |>
    g("^", "\\textasciicircum") |>
    e("%") |>
    e("$") |>
    e("#") |>
    e("_") |>
    e("{") |>
    e("}")
}

# helper function to generate a random PubMed ID
# to be used by the plugin functions
generate_pubmed <- function(amount = sample.int(10, 1)) {
  #generate 1 to 10 random integers between 1e7 and 3e7
  #this is a rough approximation of the PubMed ID range
  sample.int(30000000L, amount) + 10000000L
}

##############
# MAIN LOGIC #
##############

parse_template <- function(template_file) {
  # Read the template
  if (!file.exists(template_file)) {
    stop("Template file does not exist: ", template_file)
  }
  lines <- readLines(template_file)
  text <- paste(lines, collapse = "\n")

  #derive plugin file from template name
  plugin_file <- sub(".tex", "_plugin.R", template_file, fixed = TRUE)
  if (!file.exists(plugin_file)) {
    warning(
      "The selected template does not have a corresponding plugin file: ",
      plugin_file
    )
  } else {
    #source the plugin file
    before <- ls()
    source(plugin_file)
    loaded_functions <- setdiff(ls(), c(before, "before"))
    cat(
      "Loaded plugin functions",
      paste(loaded_functions, collapse = ", "),
      "from file", plugin_file, ".\n"
    )
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

  return(list(
    text_sections = text_sections,
    section_fields = section_fields
  ))
}

#load override template if one was provided
if (!is.na(args$template_file)) {
  tmpl_struc <- parse_template(tmpl_file)
  text_sections <- tmpl_struc$text_sections
  section_fields <- tmpl_struc$section_fields
}

# Discover all templates in the template directory
discover_all_templates <- function(template_dir) {
  tex_files <- list.files(template_dir, pattern = "\\.tex$", full.names = TRUE)
  # Filter out templates that don't have corresponding plugin files
  valid_templates <- tex_files[sapply(tex_files, function(tex_file) {
    plugin_file <- sub("\\.tex$", "_plugin.R", tex_file)
    file.exists(plugin_file)
  })]
  return(valid_templates)
}

# Get list of templates to process
templates_to_process <- if (args$all_templates || !is.na(args$template_file)) {
  if (!is.na(args$template_file)) {
    # Single template override
    list(args$template_file)
  } else {
    # All templates
    discover_all_templates(args$template_dir)
  }
} else {
  # Original behavior - use template registry
  NULL
}

# Process templates
if (!is.null(templates_to_process)) {
  # Process all specified templates with the same mock data
  for (template_file in templates_to_process) {
    template_name <- sub("\\.tex$", "", basename(template_file))
    cat("Processing template: ", template_file, "\n")
    
    # Parse the template
    tmpl_struc <- parse_template(template_file)
    text_sections <- tmpl_struc$text_sections
    section_fields <- tmpl_struc$section_fields
    
    # Process each dataset with this template
    outputs <- lapply(names(mock_data), \(uuid) {
      cat("  Processing dataset #", uuid, " with template ", template_name, "\n")
      dataset <- mock_data[[uuid]]
      
      # Perform interpolations
      inter_sections <- lapply(seq_along(text_sections), \(i) {
        section_name <- names(text_sections)[[i]]
        txt <- text_sections[[i]]
        fields <- section_fields[[i]]
        
        # If this is a regular text section:
        if (startsWith(section_name, "text_")) {
          for (j in seq_len(nrow(fields))) {
            label <- fields[j, "label"]
            marker <- paste0("\\data{", label, "}")
            if (startsWith(label, "plugin:")) {
              # derive the function name
              fname <- trimws(sub("^plugin:", "", label))
              # check that the function exists
              if (!exists(fname) || !is.function(get(fname))) {
                warning("Plugin function ", fname, " does not exist!")
                blurb <- "{\\it Missing plugin function!}"
              } else {
                # call the plugin function
                blurb <- do.call(fname, list(dataset))
              }
              txt <- sub(marker, blurb, txt, fixed = TRUE)
            } else if (!(label %in% names(dataset))) {
              cat("    Skipping unsupported label: ", label, "\n")
              txt <- sub(marker, paste0("{\\it Missing data} ", label),
                         txt, fixed = TRUE)
            } else {
              value <- tex_escape(dataset[[label]])
              txt <- sub(marker, value, txt, fixed = TRUE)
            }
          }
          txt
        } else {
          # Otherwise, if this is an iterator section:
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
                cat("    Skipping unsupported label: ", label, "\n")
                row <- sub(marker, "MISSING DATA!", row, fixed = TRUE)
              } else {
                value <- tex_escape(sds[[label]])
                row <- sub(marker, value, row, fixed = TRUE)
              }
            }
            # Add LaTeX table row terminator if this appears to be a table row
            if (grepl("&", row)) {
              row <- paste0(row, "\\\\")
            }
            rows <- c(rows, row)
          }
          paste(rows, collapse = "\n")
        }
      })
      paste0(inter_sections, collapse = "")
    }) |> setNames(names(mock_data))
    
    # Write outputs to file for this template
    for (uuid in names(outputs)) {
      outfile <- paste0(args$outprefix, template_name, "_", uuid, ".tex")
      cat(outputs[[uuid]], file = outfile)
    }
  }
} else {
  # Original behavior - match templates based on testing_laboratory
  #iterate over datasets
  outputs <- lapply(names(mock_data), \(uuid) {
    cat("Processing dataset #", uuid, "\n")
    dataset <- mock_data[[uuid]]
    #determine the correct template for this dataset
    #unless override was provided
    if (is.na(args$template_file)) {
      lab_name <- dataset$testing_laboratory
      tmpl_file <- match_template(lab_name)
      cat("Loading template: ", tmpl_file, "\n")
      #parse the template
      tmpl_struc <- parse_template(tmpl_file)
      text_sections <- tmpl_struc$text_sections
      section_fields <- tmpl_struc$section_fields
    }
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
          if (startsWith(label, "plugin:")) {
            # derive the function name
            fname <- trimws(sub("^plugin:", "", label))
            # check that the function exists
            if (!exists(fname) || !is.function(get(fname))) {
              warning("Plugin function ", fname, " does not exist!")
              blurb <- "{\\it Missing plugin function!}"
            } else {
              # call the plugin function
              blurb <- do.call(fname, list(dataset))
            }
            # # Use the plugin's blurb function if available,
            # if (exists("long_blurb") && is.function(long_blurb)) {
            #   blurb <- long_blurb(dataset$variants)
            # } else {
            #   warning("Missing blurb function from plugin for this template")
            #   blurb <- paste("No explanation is available, please contact lab.",
            #                  "This is a placeholder for the blurb.")
            # }
            txt <- sub(marker, blurb, txt, fixed = TRUE)
          # } else if (label == "summary_blurb") {
          #   blurb <- summary_blurb(dataset$variants)
          #   txt <- sub(marker, blurb, txt, fixed = "TRUE")
          } else if (!(label %in% names(dataset))) {
            cat("Skipping unsupported label: ", label, "\n")
            txt <- sub(marker, paste0("{\\it Missing data} ", label),
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
          # Add LaTeX table row terminator if this appears to be a table row
          if (grepl("&", row)) {
            row <- paste0(row, "\\\\")
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
}

cat("Done!")
