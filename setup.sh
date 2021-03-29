mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"priyanshusrivastava31@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.tomlig.toml
