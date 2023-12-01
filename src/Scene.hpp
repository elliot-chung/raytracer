#pragma once

#include <unordered_map>
#include "DisplayObject.hpp"

class Scene
{
public:
	typedef std::unordered_map<const char*, DisplayObject*> ObjectMap;
	bool addToScene(const char* name, DisplayObject* object)
	{
		if (objectMap.find(name) == objectMap.end())
		{
			objectMap[name] = object;
			return true;
		}
		return false;
	}

	bool removeFromSceen(const char* name, DisplayObject* object)
	{
		return (bool) objectMap.erase(name);
	}

	void update(ImGuiIO& io)
	{
		for (auto& object : objectMap)
		{
			object.second->update(io);
		}
	}

	inline ObjectMap& getObjects() { return objectMap; }
	
private:
	ObjectMap objectMap = {};
};