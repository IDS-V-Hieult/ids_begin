FROM node:latest
WORKDIR /app
COPY . .
#RUN export NODE_OPTIONS=--openssl-legacy-provider
RUN npm install
#RUN npm install node-sass@latest
RUN npm run build
#RUN npm uninstall --save-dev node-sass
#RUN npm install --save-dev node-sass
#RUN npm rebuild node-sass
#RUN npm audit fix
ENTRYPOINT npm start
