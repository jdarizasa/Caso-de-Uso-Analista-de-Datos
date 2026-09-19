FROM mcr.microsoft.com/devcontainers/universal:2

# Remove the stale Yarn APT repository. The ubuntugis feature runs apt update
# and otherwise fails because this repository's signing key is unavailable.
RUN set -eux; \
    for file in /etc/apt/sources.list /etc/apt/sources.list.d/*; do \
      [ -f "$file" ] || continue; \
      if grep -q "dl.yarnpkg.com/debian" "$file"; then \
        rm -f "$file"; \
      fi; \
    done; \
    rm -rf /var/lib/apt/lists/*
