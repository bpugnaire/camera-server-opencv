import asyncio
import socketio
from aiohttp import web
# create a Socket.IO server
sio = socketio.AsyncServer(logger=True,)

# wrap with ASGI application
#app = socketio.ASGIApp(sio)
app = web.Application()
sio.attach(app)

@sio.event
async def connect():
    print('connection established')

@sio.event
async def my_message(data):
    print('message received with ', data)
    await sio.emit('my response', {'response': 'my response'})

@sio.event
async def disconnect():
    print('disconnected from server')

async def main():
    await sio.connect('http://localhost:3000')
    await sio.wait()

if __name__ == '__main__':
    web.run_app(app)