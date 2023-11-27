#pragma once

#include <unordered_map>
#include "DisplayObject.hpp"

class Scene
{
public:
	bool addToScene(const char* name, DisplayObject* object)
	{
		try
		{
			auto _ = objectMap.at(name);
		}
		catch (int x)
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

	std::vector<DisplayObject*> getAllObjects()
	{
		std::vector<DisplayObject*> output(objectMap.size(), 0);
		int i = 0;
		for (const auto& [_, value] : objectMap)
			output[i++] = value;
		return output;
	}
	
private:
	std::unordered_map<const char*, DisplayObject*> objectMap;
};