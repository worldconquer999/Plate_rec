# TSP AI Engine

## Prerequisites
- Virtual environment folder name is:
    - **.venv** for python 64bit 
    - **.32venv** for python 32bit 

## Go to project
For both Ubuntu and Windows:
    ```
    cd ai-engine
    ```

## Active virtual environment
- For Ubuntu
    ```
    source $PWD/.venv(.32venv)/bin/activate
    ```
- For Windows 10/11:
    ```
    %cd%/.venv(.32venv)/Scripts/activate
    ```
## Install dependencies
    pip install -r requirements.txt
## Run
- Run
    - for Ubuntu

        ```
        set PYTHONPATH=$PWD
        python tools\encrypt_model.py
        python src\app.py
        ```
    - for Windows

        ```
        set PYTHONPATH=%cd%
        python tools\encrypt_model.py
        python src\app.py
        ```
- Call API from IP: http://localhost:8003
- Check list API (Swagger): http://localhost:8003/docs

## Build
- Copy script in `pyinstaller.txt` and run in terminal
- Change folder/file paths to compatible with your project location
- Output folder name is located in `dist` which name is same `app`