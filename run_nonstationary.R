library(cocons)

# Modified from:
# https://stackoverflow.com/a/78162082
get_current_file_path <- function() {
  this_file <- grep("^--file=", commandArgs(), value = TRUE)
  this_file <- gsub("^--file=", "", this_file)
  if (length(this_file) == 0) this_file <- rstudioapi::getSourceEditorContext()$path
  if (this_file == "") warning("Most likely running code in RStudio in a new R script window.")
  return(dirname(this_file))
}

krige_ns_grid <- function(method_dist, df_data_train, df_grid) {
  if (method_dist == "ns-aniso-tilt") {
    coco_model_list <- list(
      mean    = 0,
      std.dev = 1,
      scale   = formula( ~ 1),
      aniso   = formula( ~ 1 + s1 + s2),
      tilt    = formula( ~ 1),
      smooth  = 0.5,
      nugget  = -Inf
    )
  } else if (method_dist == "ns-aniso-tilt-mean") {
    coco_model_list <- list(
      mean    = formula( ~ 1 + s1),
      std.dev = 1,
      scale   = formula( ~ 1),
      aniso   = formula( ~ 1 + s1 + s2),
      tilt    = formula( ~ 1),
      smooth  = 0.5,
      nugget  = -Inf
    )
  } else {
    base::stop("invalid DISTRIBUTION argument")
  }

  # Create coco object
  coco_obj <- coco(
    type       = "dense",
    data       = df_data_train,
    locs       = as.matrix(df_data_train[,c("s1", "s2")]),
    z          = df_data_train$z1,
    model.list = coco_model_list,
    info  = list()
  )
  #coco_obj

  optim_coco <- cocoOptim(
    coco_obj,
    boundaries = getBoundaries(coco_obj, lower.value = -1, 32)
  )

  coco_preds <- cocoPredict(
    optim_coco,
    newdataset = df_grid,
    newlocs = as.matrix(df_grid[,c("s1", "s2")]),
    type = "pred"
  )

  data.frame(
    s1 = df_grid$s1,
    s2 = df_grid$s2,
    z1 = coco_preds$systematic + coco_preds$stochastic
  )
}

main <- function() {
  #Rscript run_nonstationary.R ns-aniso-tilt phi ppn
  #for i in {10..80..10}; do for j in {20..80..10}; do Rscript run_nonstationary.R ns-aniso-tilt $i $j; done; done
  #parallel --jobs 16 'Rscript run_nonstationary.R ns-aniso-tilt {} {}' ::: {10..80..10} ::: {20..80..10}
  args <- commandArgs(trailingOnly = TRUE)

  replicas_n <- 100
  grid_n <- 32
  method_dist <- args[1]

  folder_distribution_name <- sprintf(
    "a_run_%i_trials_exponential_model_%s_distribution_%s_phi",
    replicas_n, method_dist, args[2]
  )
  folder_experiment_name <- sprintf(
    "%s/a_run_%s",
    folder_distribution_name,
    args[3]
  )

  for (i in 1:replicas_n) {
    train_path <- sprintf("%s/true_grid_%03d_train.csv", folder_experiment_name, i)
    krige_ns_res_path <- sprintf("%s/krige_uns_grid_%03d.csv", folder_experiment_name, i)
    df_data_train <- data.table::fread(file.path(get_current_file_path(), train_path))
    df_grid <- expand.grid(s1 = seq(1, grid_n, 1), s2 = seq(1, grid_n, 1))
    
    df_krige_ns_res <- krige_ns_grid(method_dist, df_data_train, df_grid[,c("s1", "s2")])
    data.table::fwrite(df_krige_ns_res, file.path(get_current_file_path(), krige_ns_res_path))
  }
}

main()
#print(warnings())
