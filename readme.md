# AI_googletrend_tag [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

__This is a project for tagging articles with trending keywords__

## FastAPI Demo

Run app: `uvicorn main:app`<br>
Swagger UI page: `http://127.0.0.1:8000/docs`<br>
<br>
<img src="/demo/MACLR.gif">
<img src="/demo/MACLR-res.png">

## Structure
Concept: Embeds articles & keywords in to tokens then calculate the vectors similarity

<img height="500px" src="https://user-images.githubusercontent.com/71457201/194585412-8d558063-ffad-4e6c-a211-326fe96ab319.png">
 
## SystemInfo
```
Ubuntu 20.04.1 LTS
Cuda 10.1
python 3.8.5
```

## Install all the required packages.
To install all dependencies, run the following command:
```
pip install -r requirements.txt
```

## Input & output data structure
input
```python
[
{"title1":str,"content1":str},
{"title2":str,"content2":str},
...
]
```

output: tags sort by score(descent)
```python
{
"title1":{
 "tag1": score,
 "tag2": score,
 },
...
}
```

## Reference
pytrend: https://github.com/GeneralMills/pytrends<br>
MACLR: https://github.com/amzn/pecos/tree/mainline/examples/MACLR<br>
`For download pretrain model / training models, please check MACLR team's work`
