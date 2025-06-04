# Load required libraries and data
library(yaml)

# Load blurb_data from text_pieces.yml
blurb_data <- yaml.load_file("../data/text_pieces.yml")

#capitalize a word ("hello" -> "Hello")
cap <- function(txt) {
  substr(txt, 1, 1) <- toupper(substr(txt, 1, 1))
  txt
}
#convert a number to text (2 -> "two")
num2text <- function(num, one = FALSE) {
  if (num == 1 && !one) {
    "a" # says "a" instead of "one"
  }
  blurb_data$numbers[[num]]
}
#convert an amino acid's code to its full name
aaname <- function(aa) {
  blurb_data$residues[[tolower(aa)]]
}

# generates a summary text blurb for a set of variants
summary_blurb <- function(variants, suffix = "detected.") {
  #if there are no variants, we're done
  if (length(variants) == 0) {
    return(paste("No variants", suffix))
  }
  interpretations <- sapply(variants, `[[`, "interpretation")
  #rename "pathogenic" to "pathogenic variant", etc
  interpretations <- sapply(interpretations, \(iname) {
    #also, convert to lower case
    iname <- tolower(iname)
    if (!grepl("variant", iname)) {
      paste(iname, "variant")
    } else {
      iname
    }
  })
  #count how many there are of each type
  inter_table <- table(interpretations)
  #generate strings for each type/number (e.g "two pathogenic variants")
  vstrings <- sapply(names(inter_table), \(iname) {
    #translate the number to a text string (2 -> "two")
    numstr <- inter_table[[iname]] |> num2text()
    #add plural when number is greater 1
    if (inter_table[[iname]] > 1) {
      iname <- sub("variant", "variants", iname)
    }
    paste(numstr, iname)
  })
  #if there's only one type, we're done
  if (length(vstrings) == 1) {
    return(cap(paste(vstrings[[1]], suffix)))
  }
  #concatenate with commas and "and"
  paste(
    paste(vstrings[-length(vstrings)], collapse = ", "),
    "and", vstrings[[length(vstrings)]], suffix
  )
}