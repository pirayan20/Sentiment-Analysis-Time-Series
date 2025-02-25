.PHONY: run build

build:
	docker build -t streamlit-audio-app .

run:
	docker run -p 8501:8501 streamlit-audio-app
