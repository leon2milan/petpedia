# Mongo usage

This doc save mongo usage from deployment to operation.

```
cd /workspaces
wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu1804-5.0.3.tgz
tar zxvf mongodb-linux-x86_64-ubuntu1804-5.0.3.tgz
sudo mv mongodb-linux-x86_64-ubuntu1804-5.0.3 /usr/local/mongodb
export PATH=/usr/local/mongodb/bin:$PATH

sudo mkdir -p /workspaces/jiangbingyu/mongodb/mongo_log/
sudo mkdir -p /workspaces/jiangbingyu/mongodb/mongo_data
sudo chown jiangbingyu /workspaces/jiangbingyu/mongodb/mongo_log/     # 设置权限
sudo chown jiangbingyu /workspaces/jiangbingyu/mongodb/mongo_data  # 设置权限

mongod --dbpath /workspaces/jiangbingyu/mongodb/mongo_data --logpath /workspaces/jiangbingyu/mongodb/mongo_log/mongod.log --bind_ip 0.0.0.0 --bind_ip_all --fork --auth 

mongod --dbpath /workspaces/mongodb/mongo_data --logpath /workspaces/mongodb/mongo_log/mongod.log --bind_ip 0.0.0.0 --bind_ip_all --fork --auth 
```

If create admin user, need specify `dbOwner` role.
```
use qa
db.qa.insert({"name":"qa"})
db.createUser({
  user: 'xxx',  // 用户名
  pwd: 'xxxxx',  // 密码
  roles:[
      { role : "dbOwner", db : "behavior" } ,
      { role: 'read', db: 'qa'},
    ]
})
```

```
db.updateUser( "xxxx",
               {
                 roles : [
                           { role : "dbOwner", db : "behavior"  }
                         ]
                }
             )
````