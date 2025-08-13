# Gemfile for Jekyll site
source "https://rubygems.org"

# Specify Ruby version (optional but recommended)
ruby "3.1.0"

# Core Jekyll gem
gem "jekyll", "~> 4.3.0"

# Required for Jekyll 4+ on Ruby 3+
gem "webrick", "~> 1.7"

# GitHub Pages compatibility (uncomment if deploying to GitHub Pages)
# gem "github-pages", group: :jekyll_plugins

# Essential Jekyll plugins
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.15"           # RSS feed generation
  gem "jekyll-sitemap", "~> 1.4"         # XML sitemap generation
  gem "jekyll-seo-tag", "~> 2.8"         # SEO meta tags
  gem "jekyll-paginate", "~> 1.1"        # Pagination support
  gem "jekyll-archives", "~> 2.2"        # Archive pages for categories/tags
  gem "jekyll-redirect-from", "~> 0.16"  # Redirect functionality
  gem "jekyll-compose", "~> 0.12"        # Commands for creating posts/pages
end

# Optional but useful gems
group :development do
  gem "jekyll-admin", "~> 0.11"          # Web-based admin interface
end

# Performance and optimization gems
group :jekyll_plugins do
  gem "jekyll-minifier", "~> 0.1"        # Minify HTML, CSS, JS
  gem "jekyll-compress-images", "~> 1.2"  # Image compression
  gem "jekyll-last-modified-at", "~> 1.3" # Last modified dates
end

# Popular themes (uncomment the one you want)
# gem "minima", "~> 2.5"                 # Default Jekyll theme
# gem "minimal-mistakes-jekyll", "~> 4.24" # Popular theme
# gem "beautifuljekyll-theme", "~> 5.0"  # Beautiful Jekyll theme
# gem "jekyll-theme-chirpy", "~> 5.6"    # Chirpy theme
# gem "al-folio", "~> 0.8"               # Academic theme

# Development and testing gems
group :development, :test do
  gem "html-proofer", "~> 3.19"          # HTML validation
  gem "jekyll-livereload", "~> 0.2"      # Live reload during development
end

# Platform-specific gems (Windows users)
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", "~> 1.2"
  gem "tzinfo-data"
end

# Performance booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]
