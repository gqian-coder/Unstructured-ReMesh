/* functions from
    https://gitlab.kitware.com/utils/kwsys.git
*/

#include <string>

/**
 * Return path of a full filename (no trailing slashes).
 * Warning: returned path is converted to Unix slashes format.
 */
std::string GetFilenamePath(const std::string &filename);

/**
 * Return file name of a full filename (i.e. file name without path).
 */
std::string GetFilenameName(const std::string &filename);

/**
 * Return file extension of a full filename (dot included).
 * Warning: this is the longest extension (for example: .tar.gz)
 */
std::string GetFilenameExtension(const std::string &filename);

/**
 * Returns true if str1 starts (respectively ends) with str2
 */
bool StringStartsWith(const std::string &str1, const char *str2);