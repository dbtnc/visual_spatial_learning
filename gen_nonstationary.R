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

args <- commandArgs(trailingOnly = TRUE)

grid_n <- as.numeric(args[1])
coco_phi <- as.numeric(args[6]) * grid_n
coco_seed <- as.numeric(args[3])

save_path <- paste0(get_current_file_path(), "/", args[2])

x_coords <- seq(1, grid_n)
y_coords <- seq(1, grid_n)
grid <- expand.grid(x = x_coords, y = y_coords)
df_grid <- data.frame(s1 = grid[,1], s2 = grid[,2])

method_dist <- args[4]

if (method_dist == "ns-aniso") {
  coco_model_list <- list(
    mean    = 0,
    std.dev = 1,
    scale   = log(coco_phi),
    aniso   = formula( ~ 1 + s1 + s2),
    tilt    = 0,
    smooth  = 0.5,
    nugget  = -Inf
  )

  coco_sim_pars <- c(
    0.23, -0.3, 0.07
  )
} else if (method_dist == "ns-aniso-tilt") {
  coco_model_list <- list(
    mean    = 0,
    std.dev = 1,
    scale   = log(coco_phi),
    aniso   = formula( ~ 1 + s1 + s2),
    tilt    = formula( ~ 1),
    smooth  = 0.5,
    nugget  = -Inf
  )

  coco_sim_pars <- c(
    0.23, -0.3, 0.07,
    24.5 * (pi/180)
  )
} else if (method_dist == "ns-aniso-tilt-mean") {
  coco_model_list <- list(
    mean    = formula( ~ 1 + s1),
    std.dev = 1,
    scale   = log(coco_phi),
    aniso   = formula( ~ 1 + s1 + s2),
    tilt    = formula( ~ 1),
    smooth  = 0.5,
    nugget  = -Inf
  )

  coco_sim_pars <- c(
    0, 0.42,
    0.23, -0.3, 0.07,
    24.5 * (pi/180)
  )
} else {
  base::stop("invalid DISTRIBUTION argument")
}

coco_obj <- coco(
  type       = "dense",
  data       = df_grid,
  locs       = as.matrix(grid),
  z          = rep(NA, nrow(grid)),
  model.list = coco_model_list,
  info  = list()
)

sim_values <- cocoSim(
  coco.object = coco_obj,
  pars        = coco_sim_pars,
  n           = 1,
  standardize = TRUE,
  type        = "classic",
  seed        = coco_seed
)

df_grid$z1 <- sim_values[,1]
df_grid$s1 <- df_grid$s1 - 1
df_grid$s2 <- df_grid$s2 - 1

write.csv(df_grid, save_path, row.names = FALSE)
