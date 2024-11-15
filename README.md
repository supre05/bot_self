To run the app and allow for optimal performance, use command

```
gunicorn -w 2 --threads 4 -b 0.0.0.0:5000 app:app
```
