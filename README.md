# BienBaoGiaoThong
Nhận diện biển báo giao thông

First install for this project

```shell
python -m venv env
pip install opencv-python
pip install fastapi
pip install uvicorn
pip install jinja2
pip install python-multipart
```

Window

```shell
. env/Scripts/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

or using docker

```shell
docker-compose up
```


