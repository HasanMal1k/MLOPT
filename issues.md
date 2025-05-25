# Module 1: User Profiling
    Credit card service integrations
    Profile Editing
    Profile Settings Page

# Module 2: Dashboard
    Time Series Data Preprocessing and Transformation
    Notifications

# Module 3: Data Asset Provision
    Kaggle Upload

# Module 4: 
    EDA Missing

# Module 5:
    Preprocessing issue
    Transformation Out    


Landing page -> login (include credit card information) -> dashboard



<!-- Hasan Changes -->
New Homepage, 3D 


auto preprocessing = auto cleaning

data type to dataset tyoe under auto detect type


issue in custom cleaning INFO:     127.0.0.1:56852 - "POST /custom-preprocessing/preview-transformation/ HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\uvicorn\protocols\http\httptools_impl.py", line 409, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\uvicorn\middleware\proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\fastapi\applications.py", line 1054, in __call__
    await super().__call__(scope, receive, send)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\applications.py", line 112, in __call__
    await self.middleware_stack(scope, receive, send)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\middleware\errors.py", line 187, in __call__
    raise exc
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\middleware\errors.py", line 165, in __call__
    await self.app(scope, receive, _send)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\middleware\cors.py", line 93, in __call__
    await self.simple_response(scope, receive, send, request_headers=headers)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\middleware\cors.py", line 144, in simple_response
    await self.app(scope, receive, send)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\middleware\exceptions.py", line 62, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\routing.py", line 715, in __call__
    await self.middleware_stack(scope, receive, send)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\routing.py", line 735, in app
    await route.handle(scope, receive, send)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\routing.py", line 288, in handle
    await self.app(scope, receive, send)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\routing.py", line 76, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\routing.py", line 73, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\fastapi\routing.py", line 338, in app
    response = actual_response_class(content, **response_args)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\responses.py", line 181, in __init__
    super().__init__(content, status_code, headers, media_type, background)
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\responses.py", line 44, in __init__
    self.body = self.render(content)
                ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\mhasa\Desktop\MLOPT\server\Lib\site-packages\starlette\responses.py", line 184, in render
    return json.dumps(
           ^^^^^^^^^^^
  File "C:\Users\mhasa\AppData\Local\Programs\Python\Python312\Lib\json\__init__.py", line 238, in dumps
    **kw).encode(obj)
          ^^^^^^^^^^^
  File "C:\Users\mhasa\AppData\Local\Programs\Python\Python312\Lib\json\encoder.py", line 200, in encode
    chunks = self.iterencode(o, _one_shot=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\mhasa\AppData\Local\Programs\Python\Python312\Lib\json\encoder.py", line 258, in iterencode
    return _iterencode(o, 0)
           ^^^^^^^^^^^^^^^^^
ValueError: Out of range float values are not JSON compliant: nan





Kaggle import issue