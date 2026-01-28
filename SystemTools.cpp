#include "SystemTools.hpp"
#include <cstring>

/**
 * Return path of a full filename (no trailing slashes).
 * Warning: returned path is converted to Unix slashes format.
 */
std::string GetFilenamePath(const std::string &filename)
{
    std::string fn = filename;

    std::string::size_type slash_pos = fn.rfind('/');
    if (slash_pos == 0)
    {
        return "/";
    }
    if (slash_pos == 2 && fn[1] == ':')
    {
        // keep the / after a drive letter
        fn.resize(3);
        return fn;
    }
    if (slash_pos == std::string::npos)
    {
        return "";
    }
    fn.resize(slash_pos);
    return fn;
}

/**
 * Return file name of a full filename (i.e. file name without path).
 */
std::string GetFilenameName(const std::string &filename)
{
    char separators = '/';
    std::string::size_type slash_pos = filename.find_last_of(separators);
    if (slash_pos != std::string::npos)
    {
        return filename.substr(slash_pos + 1);
    }
    else
    {
        return filename;
    }
}

/**
 * Return file extension of a full filename (dot included).
 * Warning: this is the longest extension (for example: .tar.gz)
 */
std::string GetFilenameExtension(const std::string &filename)
{
    std::string name = GetFilenameName(filename);
    std::string::size_type dot_pos = name.find('.');
    if (dot_pos != std::string::npos)
    {
        name.erase(0, dot_pos);
        return name;
    }
    else
    {
        return "";
    }
}

// Returns if string starts with another string
bool StringStartsWith(const std::string &str1, const char *str2)
{
    if (!str2)
    {
        return false;
    }
    size_t len1 = str1.size(), len2 = strlen(str2);
    return len1 >= len2 && !strncmp(str1.c_str(), str2, len2) ? true : false;
}