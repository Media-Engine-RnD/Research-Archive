# Gemfile for Jekyll 4.4.1 site
source "https://rubygems.org"

# Core Jekyll
gem "jekyll", "~> 4.4.1"
gem "webrick", "~> 1.8"

# Plugins compatible with Jekyll 4.x
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.17"              # RSS feed generation
  gem "jekyll-sitemap", "~> 1.4"            # XML sitemap
  gem "jekyll-seo-tag", "~> 2.8"            # SEO meta tags
  gem "jekyll-paginate", "~> 1.1"           # Pagination
  gem "jekyll-archives", "~> 2.2"           # Archives by category/tag
  gem "jekyll-redirect-from", "~> 0.16"     # Redirects
  gem "jekyll-compose", "~> 0.12"           # Post/page creation commands
  gem "jekyll-minifier", "~> 0.0.1"         # Minify HTML, CSS, JS
  gem "jekyll-compress-images", "~> 1.2"    # Image compression
  gem "jekyll-last-modified-at", "~> 1.3"   # Last modified date stamps
  gem "jekyll-remote-theme", "~> 0.4.3"     # Remote theme support
end

# Development tools
group :development do
  gem "jekyll-admin", "~> 0.11"             # Web-based admin UI
end

# Testing
group :development, :test do
  gem "html-proofer", "~> 3.19"             # Validate HTML output
end

# Windows-specific gems
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", "~> 1.2"
  gem "tzinfo-data"
end
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]
