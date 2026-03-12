library(sp)
library(gstat)

# Modified from:
# https://stackoverflow.com/a/78162082
get_current_file_path <- function() {
  this_file <- grep("^--file=", commandArgs(), value = TRUE)
  this_file <- gsub("^--file=", "", this_file)
  if (length(this_file) == 0) this_file <- rstudioapi::getSourceEditorContext()$path
  if (this_file == "") warning("Most likely running code in RStudio in a new R script window.")
  return(dirname(this_file))
}

krige_grid <- function(sp_data, sp_grid, formula, model = vgm(, "Exp"), formula_grid = ~s1+s2) {
  coordinates(sp_data) <- formula_grid
  gridded(sp_grid) <- formula_grid

  vgm1 <- variogram(formula, sp_data)
  m <- fit.variogram(vgm1, model)
  # ordinary kriging:
  krige_res <- krige(formula, sp_data, sp_grid, model = m, debug.level = 0)
}

error_warn_fn <- function(err_or_warn) {
  args <- commandArgs(trailingOnly = TRUE)
  current_iter <- strsplit(tail(strsplit(args[1], "_")[[1]], n = 1), ".", fixed = TRUE)[[1]][1]
  cat(
    paste0(current_iter, " - ", args[7], "\n"),
    file = args[6],
    append = TRUE
  )
}

main <- function() {
  tryCatch(
    expr = {
      set.seed(123)

      args <- commandArgs(trailingOnly = TRUE)

      sp_data <- read.csv(paste0(get_current_file_path(), "/", args[1]))
      sp_data$s1 <- sp_data$s1 + 1
      sp_data$s2 <- sp_data$s2 + 1

      # Grid size
      size <- sqrt(nrow(sp_data))
      # n points
      n <- round((size^2) * as.numeric(args[5]))
      sp_grid <- expand.grid(s1 = seq(1, size, 1), s2 = seq(1, size, 1))

      ids_sample <- sample(nrow(sp_data), n)
      ids_train <- sample(ids_sample, round(0.7 * length(ids_sample)))
      ids_test <- setdiff(ids_sample, ids_train)

      sp_data_train <- sp_data[ids_train,]
      sp_data_test <- sp_data[ids_test,]

      if (as.numeric(args[7]) > 50000) {
        krige_res <- data.frame(
          s1 = sp_data$s1,
          s2 = sp_data$s2,
          var1.pred = Inf,
          var1.var = Inf
        )
      } else {
        krige_res <- krige_grid(sp_data_train, sp_grid, z1 ~ 1)
      }

      current_iter <- strsplit(tail(strsplit(args[1], "_")[[1]], n = 1), ".", fixed = TRUE)[[1]][1]

      write.csv(sp_data_train, paste0(get_current_file_path(), "/", args[2]), row.names = FALSE)
      write.csv(sp_data_test, paste0(get_current_file_path(), "/", args[3]), row.names = FALSE)
      write.csv(as.data.frame(krige_res), paste0(get_current_file_path(), "/", args[4]), row.names = FALSE)
      cat(paste("Finished Krige:", current_iter, "\n"))
    },
    error = error_warn_fn,
    warning = error_warn_fn
  )
}

main()
