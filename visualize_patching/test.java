static void reduceTransitive(Collection<GroupTransition> transitive)
    {
        final Collection<GroupTransition> localChecked = new ArrayList<>(transitive.size());
        final Collection<GroupTransition> toRemove = new ArrayList<>();

        final Iterator<GroupTransition> iterator = transitive.iterator();
        GroupTransition first = iterator.next();
        while (iterator.hasNext())
        {
            final GroupTransition current = iterator.next();
            localChecked.add(current);
            if (current.getOut().equals(first.getOut()))
            {
                toRemove.addAll(localChecked);

                localChecked.clear();

                if (iterator.hasNext())
                {
                    first = iterator.next();
                }
            }
        }
        transitive.removeAll(toRemove);
        toRemove.clear();
    }