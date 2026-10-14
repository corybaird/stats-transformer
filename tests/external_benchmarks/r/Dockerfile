FROM r-base:4.4.2
RUN apt-get update && apt-get install -y --no-install-recommends r-cran-urca r-cran-plm r-cran-aer r-cran-sandwich r-cran-lmtest r-cran-strucchange r-cran-mass r-cran-jsonlite ca-certificates && rm -rf /var/lib/apt/lists/*
RUN Rscript -e 'options(warn=2, timeout=600); install.packages("vars", repos="https://cloud.r-project.org/", dependencies=FALSE)'
WORKDIR /repo
